import logging
import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from decimal import Decimal

from config.config_manager import ConfigManager
from core.services.exchange_interface import ExchangeInterface
from core.bot_management.event_bus import EventBus, Events
from core.order_handling.futures_position_manager import FuturesPositionManager, Position, PositionEvents
from core.order_handling.futures_risk_manager import FuturesRiskManager, RiskEvents

class StopLossEvents:
    """
    Defines stop-loss-related event types for the EventBus.
    """
    STOP_LOSS_TRIGGERED = "stop_loss_triggered"
    STOP_LOSS_WARNING = "stop_loss_warning"
    STOP_LOSS_EXECUTED = "stop_loss_executed"
    STOP_LOSS_SETTINGS_UPDATED = "stop_loss_settings_updated"
    STOP_LOSS_METRICS_UPDATE = "stop_loss_metrics_update"
    EXTERNAL_SIGNAL_STOP_LOSS = "external_signal_stop_loss"

class StopLossManager:
    """
    Manages stop loss mechanisms for futures trading operations.
    
    Responsibilities:
    1. USDT-based stop loss implementation
    2. Portfolio percentage-based stop loss implementation
    3. Real-time loss monitoring
    4. Position closing execution
    5. Integration with the FuturesPositionManager and FuturesRiskManager
    6. Event publishing for stop loss events
    """
    
    def __init__(
        self,
        config_manager: ConfigManager,
        exchange_service: ExchangeInterface,
        event_bus: EventBus,
        position_manager: FuturesPositionManager,
        risk_manager: Optional[FuturesRiskManager] = None
    ):
        """
        Initialize the StopLossManager.
        
        Args:
            config_manager: Configuration manager instance
            exchange_service: Exchange service instance
            event_bus: Event bus for publishing stop loss events
            position_manager: Futures position manager instance
            risk_manager: Optional futures risk manager instance for coordination
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config_manager = config_manager
        self.exchange_service = exchange_service
        self.event_bus = event_bus
        self.position_manager = position_manager
        self.risk_manager = risk_manager
        
        # Validate that we're in futures mode
        if not self.config_manager.is_futures_market():
            raise ValueError("StopLossManager can only be used with futures markets")
        
        # Initialize stop loss tracking
        self.stop_loss_metrics = {}
        self.stop_loss_update_tasks = set()
        self.stop_loss_active = False
        self.stop_loss_history = []
        
        # Get trading pair
        self.base_currency = self.config_manager.get_base_currency()
        self.quote_currency = self.config_manager.get_quote_currency()
        self.pair = f"{self.base_currency}/{self.quote_currency}"
        self.exchange_pair = self.pair.replace("/", "")
        
        # Load stop loss settings
        self._load_stop_loss_settings()
        
        self.logger.info(f"Initialized StopLossManager for {self.pair}")
    
    def _load_stop_loss_settings(self) -> None:
        """
        Load stop loss settings from configuration.
        """
        # Get risk management settings
        risk_management = self.config_manager.get_risk_management()
        futures_risk = self.config_manager.get_futures_risk_management()
        
        # Load standard stop loss settings
        stop_loss = self.config_manager.get_stop_loss()
        self.standard_stop_loss_enabled = self.config_manager.is_stop_loss_enabled()
        self.standard_stop_loss_threshold = self.config_manager.get_stop_loss_threshold()
        
        # Initialize stop loss settings with defaults
        self.stop_loss_settings = {
            # USDT-based stop loss
            "usdt_stop_loss": {
                "enabled": futures_risk.get("usdt_stop_loss", {}).get("enabled", False),
                "max_loss_amount": futures_risk.get("usdt_stop_loss", {}).get("max_loss_amount", 1000.0),  # Default 1000 USDT
                "per_position": futures_risk.get("usdt_stop_loss", {}).get("per_position", False),
                "warning_threshold": futures_risk.get("usdt_stop_loss", {}).get("warning_threshold", 0.7)  # 70% of max loss
            },
            
            # Portfolio percentage-based stop loss
            "portfolio_stop_loss": {
                "enabled": futures_risk.get("portfolio_stop_loss", {}).get("enabled", False),
                "max_loss_percentage": futures_risk.get("portfolio_stop_loss", {}).get("max_loss_percentage", 0.1),  # 10% of portfolio
                "warning_threshold": futures_risk.get("portfolio_stop_loss", {}).get("warning_threshold", 0.7)  # 70% of max loss
            },
            
            # Trailing stop loss
            "trailing_stop_loss": {
                "enabled": futures_risk.get("trailing_stop_loss", {}).get("enabled", False),
                "activation_threshold": futures_risk.get("trailing_stop_loss", {}).get("activation_threshold", 0.02),  # 2% profit
                "trailing_distance": futures_risk.get("trailing_stop_loss", {}).get("trailing_distance", 0.01)  # 1% trailing distance
            },
            
            # External signal integration
            "external_signals": {
                "enabled": futures_risk.get("external_signals", {}).get("enabled", False),
                "signal_sources": futures_risk.get("external_signals", {}).get("signal_sources", [])
            },
            
            # General settings
            "monitoring_interval": futures_risk.get("stop_loss_monitoring_interval", 5),  # Check every 5 seconds
            "execution_type": futures_risk.get("stop_loss_execution_type", "market"),  # Market or limit order
            "partial_close_enabled": futures_risk.get("partial_close_enabled", False),
            "partial_close_percentage": futures_risk.get("partial_close_percentage", 0.5)  # Close 50% of position
        }
        
        self.logger.info(f"Loaded stop loss settings: USDT stop loss: {self.stop_loss_settings['usdt_stop_loss']['enabled']}, "
                         f"Portfolio stop loss: {self.stop_loss_settings['portfolio_stop_loss']['enabled']}")
    
    async def initialize(self) -> None:
        """
        Initialize the stop loss manager and start monitoring tasks.
        """
        try:
            # Initialize stop loss metrics
            await self._initialize_stop_loss_metrics()
            
            # Start stop loss monitoring tasks
            self._start_stop_loss_monitoring()
            
            # Subscribe to position events
            await self._subscribe_to_position_events()
            
            # Subscribe to risk events if risk manager is provided
            if self.risk_manager:
                await self._subscribe_to_risk_events()
            
            self.logger.info(f"StopLossManager initialized successfully for {self.pair}")
        except Exception as e:
            self.logger.error(f"Error initializing StopLossManager: {e}")
            raise
    
    async def _initialize_stop_loss_metrics(self) -> None:
        """
        Initialize stop loss metrics with current market data.
        """
        try:
            # Get account balance
            balance_data = await self.exchange_service.get_balance()
            total_balance = float(balance_data.get(self.quote_currency, {}).get('total', 0))
            
            # Get current positions
            positions = await self.position_manager.get_all_positions()
            
            # Calculate initial portfolio value
            unrealized_pnl = sum(position.get('unrealized_pnl', 0) for position in positions)
            portfolio_value = total_balance + unrealized_pnl
            
            # Initialize stop loss metrics
            self.stop_loss_metrics = {
                "initial_portfolio_value": portfolio_value,
                "current_portfolio_value": portfolio_value,
                "max_portfolio_value": portfolio_value,
                "total_realized_pnl": 0.0,
                "total_unrealized_pnl": unrealized_pnl,
                "usdt_loss_threshold": self.stop_loss_settings["usdt_stop_loss"]["max_loss_amount"],
                "portfolio_loss_threshold": portfolio_value * self.stop_loss_settings["portfolio_stop_loss"]["max_loss_percentage"],
                "current_usdt_loss": 0.0,
                "current_portfolio_loss_percentage": 0.0,
                "stop_loss_triggers_count": 0,
                "last_stop_loss_time": 0,
                "positions_with_stop_loss": {}
            }
            
            # Initialize position-specific stop loss metrics
            for position in positions:
                position_id = position.get('position_id', f"{position.get('pair')}-{position.get('position_side')}")
                self.stop_loss_metrics["positions_with_stop_loss"][position_id] = {
                    "entry_price": position.get('entry_price', 0),
                    "size": position.get('size', 0),
                    "initial_value": position.get('size', 0) * position.get('entry_price', 0),
                    "current_value": position.get('size', 0) * position.get('entry_price', 0),
                    "max_value": position.get('size', 0) * position.get('entry_price', 0),
                    "unrealized_pnl": position.get('unrealized_pnl', 0),
                    "stop_loss_price": 0.0,
                    "stop_loss_triggered": False
                }
            
            self.logger.info(f"Initialized stop loss metrics with portfolio value: {portfolio_value}")
        except Exception as e:
            self.logger.error(f"Error initializing stop loss metrics: {e}")
            raise
    
    async def _subscribe_to_position_events(self) -> None:
        """
        Subscribe to position-related events from the position manager.
        """
        try:
            # Subscribe to position opened events
            await self.event_bus.subscribe(PositionEvents.POSITION_OPENED, self._handle_position_opened)
            
            # Subscribe to position modified events
            await self.event_bus.subscribe(PositionEvents.POSITION_MODIFIED, self._handle_position_modified)
            
            # Subscribe to position closed events
            await self.event_bus.subscribe(PositionEvents.POSITION_CLOSED, self._handle_position_closed)
            
            # Subscribe to position PnL update events
            await self.event_bus.subscribe(PositionEvents.POSITION_PNL_UPDATE, self._handle_position_pnl_update)
            
            self.logger.info("Subscribed to position events")
        except Exception as e:
            self.logger.error(f"Error subscribing to position events: {e}")
            raise
    
    async def _subscribe_to_risk_events(self) -> None:
        """
        Subscribe to risk-related events from the risk manager.
        """
        try:
            # Subscribe to liquidation risk events
            await self.event_bus.subscribe(RiskEvents.LIQUIDATION_RISK_DETECTED, self._handle_liquidation_risk)
            
            # Subscribe to circuit breaker events
            await self.event_bus.subscribe(RiskEvents.CIRCUIT_BREAKER_TRIGGERED, self._handle_circuit_breaker)
            
            # Subscribe to drawdown events
            await self.event_bus.subscribe(RiskEvents.DRAWDOWN_THRESHOLD_EXCEEDED, self._handle_drawdown_exceeded)
            
            self.logger.info("Subscribed to risk events")
        except Exception as e:
            self.logger.error(f"Error subscribing to risk events: {e}")
            raise
    
    def _start_stop_loss_monitoring(self) -> None:
        """
        Start background tasks for monitoring various stop loss conditions.
        """
        # Start USDT-based stop loss monitoring
        if self.stop_loss_settings["usdt_stop_loss"]["enabled"]:
            usdt_task = asyncio.create_task(self._monitor_usdt_stop_loss())
            self.stop_loss_update_tasks.add(usdt_task)
            usdt_task.add_done_callback(self.stop_loss_update_tasks.discard)
        
        # Start portfolio percentage-based stop loss monitoring
        if self.stop_loss_settings["portfolio_stop_loss"]["enabled"]:
            portfolio_task = asyncio.create_task(self._monitor_portfolio_stop_loss())
            self.stop_loss_update_tasks.add(portfolio_task)
            portfolio_task.add_done_callback(self.stop_loss_update_tasks.discard)
        
        # Start trailing stop loss monitoring
        if self.stop_loss_settings["trailing_stop_loss"]["enabled"]:
            trailing_task = asyncio.create_task(self._monitor_trailing_stop_loss())
            self.stop_loss_update_tasks.add(trailing_task)
            trailing_task.add_done_callback(self.stop_loss_update_tasks.discard)
        
        # Start external signal monitoring
        if self.stop_loss_settings["external_signals"]["enabled"]:
            signal_task = asyncio.create_task(self._monitor_external_signals())
            self.stop_loss_update_tasks.add(signal_task)
            signal_task.add_done_callback(self.stop_loss_update_tasks.discard)
        
        # Start stop loss metrics reporting
        metrics_task = asyncio.create_task(self._report_stop_loss_metrics())
        self.stop_loss_update_tasks.add(metrics_task)
        metrics_task.add_done_callback(self.stop_loss_update_tasks.discard)
        
        self.logger.info("Started stop loss monitoring tasks")
    
    async def _monitor_usdt_stop_loss(self) -> None:
        """
        Monitor positions for USDT-based stop loss conditions.
        """
        try:
            usdt_stop_loss_settings = self.stop_loss_settings["usdt_stop_loss"]
            monitoring_interval = self.stop_loss_settings["monitoring_interval"]
            
            while True:
                # Skip if USDT stop loss is disabled
                if not usdt_stop_loss_settings["enabled"]:
                    await asyncio.sleep(monitoring_interval)
                    continue
                
                # Get all positions
                positions = await self.position_manager.get_all_positions()
                
                # Calculate total unrealized PnL (loss only)
                total_unrealized_loss = 0.0
                for position in positions:
                    unrealized_pnl = position.get('unrealized_pnl', 0)
                    if unrealized_pnl < 0:
                        total_unrealized_loss += abs(unrealized_pnl)
                
                # Update stop loss metrics
                self.stop_loss_metrics["current_usdt_loss"] = total_unrealized_loss
                
                # Check if total loss exceeds the threshold
                max_loss_amount = usdt_stop_loss_settings["max_loss_amount"]
                warning_threshold = usdt_stop_loss_settings["warning_threshold"] * max_loss_amount
                
                # Check for per-position stop loss if enabled
                if usdt_stop_loss_settings["per_position"]:
                    for position in positions:
                        position_id = position.get('position_id', f"{position.get('pair')}-{position.get('position_side')}")
                        unrealized_pnl = position.get('unrealized_pnl', 0)
                        
                        # Skip positions with positive PnL
                        if unrealized_pnl >= 0:
                            continue
                        
                        position_loss = abs(unrealized_pnl)
                        
                        # Check if position loss exceeds warning threshold
                        if position_loss > warning_threshold and position_loss < max_loss_amount:
                            await self._handle_stop_loss_warning(
                                position=position,
                                current_loss=position_loss,
                                max_loss=max_loss_amount,
                                stop_loss_type="usdt_per_position"
                            )
                        
                        # Check if position loss exceeds max loss amount
                        if position_loss >= max_loss_amount:
                            await self._execute_stop_loss(
                                position=position,
                                reason=f"USDT-based stop loss triggered: Position loss {position_loss:.2f} {self.quote_currency} exceeds threshold {max_loss_amount:.2f} {self.quote_currency}",
                                stop_loss_type="usdt_per_position"
                            )
                else:
                    # Check if total loss exceeds warning threshold
                    if total_unrealized_loss > warning_threshold and total_unrealized_loss < max_loss_amount:
                        # Publish warning event
                        await self.event_bus.publish(
                            StopLossEvents.STOP_LOSS_WARNING,
                            {
                                "stop_loss_type": "usdt_total",
                                "current_loss": total_unrealized_loss,
                                "max_loss": max_loss_amount,
                                "warning_threshold": warning_threshold,
                                "positions": positions
                            }
                        )
                        
                        self.logger.warning(
                            f"USDT stop loss warning: Current loss {total_unrealized_loss:.2f} {self.quote_currency} "
                            f"approaching threshold {max_loss_amount:.2f} {self.quote_currency}"
                        )
                    
                    # Check if total loss exceeds max loss amount
                    if total_unrealized_loss >= max_loss_amount:
                        # Trigger stop loss for all positions with negative PnL
                        for position in positions:
                            unrealized_pnl = position.get('unrealized_pnl', 0)
                            if unrealized_pnl < 0:
                                await self._execute_stop_loss(
                                    position=position,
                                    reason=f"USDT-based stop loss triggered: Total loss {total_unrealized_loss:.2f} {self.quote_currency} exceeds threshold {max_loss_amount:.2f} {self.quote_currency}",
                                    stop_loss_type="usdt_total"
                                )
                
                # Check every monitoring interval
                await asyncio.sleep(monitoring_interval)
        except asyncio.CancelledError:
            self.logger.info("USDT stop loss monitoring task cancelled")
        except Exception as e:
            self.logger.error(f"Error in USDT stop loss monitoring: {e}")
    
    async def _monitor_portfolio_stop_loss(self) -> None:
        """
        Monitor portfolio for percentage-based stop loss conditions.
        """
        try:
            portfolio_stop_loss_settings = self.stop_loss_settings["portfolio_stop_loss"]
            monitoring_interval = self.stop_loss_settings["monitoring_interval"]
            
            while True:
                # Skip if portfolio stop loss is disabled
                if not portfolio_stop_loss_settings["enabled"]:
                    await asyncio.sleep(monitoring_interval)
                    continue
                
                # Get account balance
                balance_data = await self.exchange_service.get_balance()
                total_balance = float(balance_data.get(self.quote_currency, {}).get('total', 0))
                
                # Get all positions
                positions = await self.position_manager.get_all_positions()
                
                # Calculate unrealized PnL
                unrealized_pnl = sum(position.get('unrealized_pnl', 0) for position in positions)
                
                # Calculate current portfolio value
                current_portfolio_value = total_balance + unrealized_pnl
                
                # Update max portfolio value if current value is higher
                if current_portfolio_value > self.stop_loss_metrics["max_portfolio_value"]:
                    self.stop_loss_metrics["max_portfolio_value"] = current_portfolio_value
                
                # Calculate drawdown from max portfolio value
                max_portfolio_value = self.stop_loss_metrics["max_portfolio_value"]
                initial_portfolio_value = self.stop_loss_metrics["initial_portfolio_value"]
                
                # Calculate loss percentage from initial value and from max value
                loss_from_initial = 0.0
                if initial_portfolio_value > 0:
                    loss_from_initial = max(0, (initial_portfolio_value - current_portfolio_value) / initial_portfolio_value)
                
                loss_from_max = 0.0
                if max_portfolio_value > 0:
                    loss_from_max = max(0, (max_portfolio_value - current_portfolio_value) / max_portfolio_value)
                
                # Use the larger of the two loss percentages
                portfolio_loss_percentage = max(loss_from_initial, loss_from_max)
                
                # Update stop loss metrics
                self.stop_loss_metrics["current_portfolio_value"] = current_portfolio_value
                self.stop_loss_metrics["current_portfolio_loss_percentage"] = portfolio_loss_percentage
                
                # Get max loss percentage threshold
                max_loss_percentage = portfolio_stop_loss_settings["max_loss_percentage"]
                warning_threshold = portfolio_stop_loss_settings["warning_threshold"] * max_loss_percentage
                
                # Check if loss percentage exceeds warning threshold
                if portfolio_loss_percentage > warning_threshold and portfolio_loss_percentage < max_loss_percentage:
                    # Publish warning event
                    await self.event_bus.publish(
                        StopLossEvents.STOP_LOSS_WARNING,
                        {
                            "stop_loss_type": "portfolio_percentage",
                            "current_loss_percentage": portfolio_loss_percentage,
                            "max_loss_percentage": max_loss_percentage,
                            "warning_threshold": warning_threshold,
                            "initial_portfolio_value": initial_portfolio_value,
                            "max_portfolio_value": max_portfolio_value,
                            "current_portfolio_value": current_portfolio_value,
                            "positions": positions
                        }
                    )
                    
                    self.logger.warning(
                        f"Portfolio stop loss warning: Current loss {portfolio_loss_percentage:.2%} "
                        f"approaching threshold {max_loss_percentage:.2%}"
                    )
                
                # Check if loss percentage exceeds max loss percentage
                if portfolio_loss_percentage >= max_loss_percentage:
                    # Trigger stop loss for all positions
                    for position in positions:
                        await self._execute_stop_loss(
                            position=position,
                            reason=f"Portfolio-based stop loss triggered: Loss {portfolio_loss_percentage:.2%} exceeds threshold {max_loss_percentage:.2%}",
                            stop_loss_type="portfolio_percentage"
                        )
                
                # Check every monitoring interval
                await asyncio.sleep(monitoring_interval)
        except asyncio.CancelledError:
            self.logger.info("Portfolio stop loss monitoring task cancelled")
        except Exception as e:
            self.logger.error(f"Error in portfolio stop loss monitoring: {e}")
    
    async def _monitor_trailing_stop_loss(self) -> None:
        """
        Monitor positions for trailing stop loss conditions.
        """
        try:
            trailing_stop_loss_settings = self.stop_loss_settings["trailing_stop_loss"]
            monitoring_interval = self.stop_loss_settings["monitoring_interval"]
            
            # Dictionary to track trailing stop levels for each position
            trailing_stops = {}
            
            while True:
                # Skip if trailing stop loss is disabled
                if not trailing_stop_loss_settings["enabled"]:
                    await asyncio.sleep(monitoring_interval)
                    continue
                
                # Get all positions
                positions = await self.position_manager.get_all_positions()
                
                for position in positions:
                    position_id = position.get('position_id', f"{position.get('pair')}-{position.get('position_side')}")
                    position_side = position.get('position_side')
                    entry_price = position.get('entry_price', 0)
                    unrealized_pnl = position.get('unrealized_pnl', 0)
                    size = position.get('size', 0)
                    
                    # Skip positions with zero size
                    if size == 0:
                        continue
                    
                    # Get current price
                    current_price = await self.exchange_service.get_current_price(self.exchange_pair)
                    
                    # Calculate profit percentage
                    profit_percentage = 0.0
                    if entry_price > 0:
                        if position_side == 'long':
                            profit_percentage = (current_price - entry_price) / entry_price
                        else:
                            profit_percentage = (entry_price - current_price) / entry_price
                    
                    # Initialize trailing stop if not already set and profit exceeds activation threshold
                    activation_threshold = trailing_stop_loss_settings["activation_threshold"]
                    trailing_distance = trailing_stop_loss_settings["trailing_distance"]
                    
                    if position_id not in trailing_stops and profit_percentage >= activation_threshold:
                        # Set initial trailing stop level
                        if position_side == 'long':
                            trailing_stop_price = current_price * (1 - trailing_distance)
                        else:
                            trailing_stop_price = current_price * (1 + trailing_distance)
                        
                        trailing_stops[position_id] = {
                            "stop_price": trailing_stop_price,
                            "highest_price": current_price if position_side == 'long' else 0,
                            "lowest_price": current_price if position_side == 'short' else float('inf'),
                            "activated_at": time.time(),
                            "activated_price": current_price
                        }
                        
                        self.logger.info(
                            f"Trailing stop activated for {position_id} at price {current_price}. "
                            f"Initial stop price: {trailing_stop_price}"
                        )
                    
                    # Update trailing stop if already set
                    elif position_id in trailing_stops:
                        trailing_data = trailing_stops[position_id]
                        
                        # Update highest/lowest price
                        if position_side == 'long':
                            if current_price > trailing_data["highest_price"]:
                                # Update highest price and trailing stop
                                old_stop = trailing_data["stop_price"]
                                trailing_data["highest_price"] = current_price
                                trailing_data["stop_price"] = current_price * (1 - trailing_distance)
                                
                                self.logger.info(
                                    f"Updated trailing stop for {position_id}: New high {current_price}, "
                                    f"stop moved from {old_stop} to {trailing_data['stop_price']}"
                                )
                        else:  # short position
                            if current_price < trailing_data["lowest_price"]:
                                # Update lowest price and trailing stop
                                old_stop = trailing_data["stop_price"]
                                trailing_data["lowest_price"] = current_price
                                trailing_data["stop_price"] = current_price * (1 + trailing_distance)
                                
                                self.logger.info(
                                    f"Updated trailing stop for {position_id}: New low {current_price}, "
                                    f"stop moved from {old_stop} to {trailing_data['stop_price']}"
                                )
                        
                        # Check if price hit the trailing stop
                        if (position_side == 'long' and current_price <= trailing_data["stop_price"]) or \
                           (position_side == 'short' and current_price >= trailing_data["stop_price"]):
                            # Trailing stop hit, execute stop loss
                            await self._execute_stop_loss(
                                position=position,
                                reason=f"Trailing stop loss triggered at {current_price} (stop level: {trailing_data['stop_price']})",
                                stop_loss_type="trailing"
                            )
                            
                            # Remove trailing stop after execution
                            del trailing_stops[position_id]
                
                # Check every monitoring interval
                await asyncio.sleep(monitoring_interval)
        except asyncio.CancelledError:
            self.logger.info("Trailing stop loss monitoring task cancelled")
        except Exception as e:
            self.logger.error(f"Error in trailing stop loss monitoring: {e}")
    
    async def _monitor_external_signals(self) -> None:
        """
        Monitor for external stop loss signals.
        This is a placeholder for future integration with external signals.
        """
        try:
            external_signals_settings = self.stop_loss_settings["external_signals"]
            monitoring_interval = self.stop_loss_settings["monitoring_interval"]
            
            while True:
                # Skip if external signals are disabled
                if not external_signals_settings["enabled"]:
                    await asyncio.sleep(monitoring_interval)
                    continue
                
                # This would be implemented based on the specific external signal sources
                # For now, it's just a placeholder
                
                # Check every monitoring interval
                await asyncio.sleep(monitoring_interval)
        except asyncio.CancelledError:
            self.logger.info("External signals monitoring task cancelled")
        except Exception as e:
            self.logger.error(f"Error in external signals monitoring: {e}")
    
    async def _report_stop_loss_metrics(self) -> None:
        """
        Periodically report stop loss metrics to subscribers.
        """
        try:
            while True:
                # Publish stop loss metrics update event
                await self.event_bus.publish(
                    StopLossEvents.STOP_LOSS_METRICS_UPDATE,
                    {
                        "pair": self.pair,
                        "metrics": self.stop_loss_metrics,
                        "settings": self.stop_loss_settings,
                        "timestamp": time.time()
                    }
                )
                
                # Report every 5 minutes
                await asyncio.sleep(300)
        except asyncio.CancelledError:
            self.logger.info("Stop loss metrics reporting task cancelled")
        except Exception as e:
            self.logger.error(f"Error in stop loss metrics reporting: {e}")
    
    async def _handle_stop_loss_warning(self, position: Dict[str, Any], current_loss: float, max_loss: float, stop_loss_type: str) -> None:
        """
        Handle stop loss warning by publishing an event.
        
        Args:
            position: Position data
            current_loss: Current loss amount or percentage
            max_loss: Maximum loss threshold
            stop_loss_type: Type of stop loss (usdt, portfolio, trailing)
        """
        try:
            position_id = position.get('position_id', f"{position.get('pair')}-{position.get('position_side')}")
            
            # Publish warning event
            await self.event_bus.publish(
                StopLossEvents.STOP_LOSS_WARNING,
                {
                    "stop_loss_type": stop_loss_type,
                    "position_id": position_id,
                    "position": position,
                    "current_loss": current_loss,
                    "max_loss": max_loss,
                    "warning_percentage": current_loss / max_loss if max_loss > 0 else 1.0
                }
            )
            
            self.logger.warning(
                f"Stop loss warning for {position_id}: Current loss {current_loss:.2f} "
                f"approaching threshold {max_loss:.2f} ({stop_loss_type})"
            )
        except Exception as e:
            self.logger.error(f"Error handling stop loss warning: {e}")
    
    async def _execute_stop_loss(self, position: Dict[str, Any], reason: str, stop_loss_type: str) -> None:
        """
        Execute stop loss by closing the position.
        
        Args:
            position: Position data
            reason: Reason for stop loss execution
            stop_loss_type: Type of stop loss (usdt, portfolio, trailing)
        """
        try:
            position_id = position.get('position_id', f"{position.get('pair')}-{position.get('position_side')}")
            position_side = position.get('position_side')
            
            # Check if partial close is enabled
            partial_close_enabled = self.stop_loss_settings["partial_close_enabled"]
            partial_close_percentage = self.stop_loss_settings["partial_close_percentage"]
            
            # Determine order type
            order_type = self.stop_loss_settings["execution_type"]
            
            # Log stop loss execution
            self.logger.warning(
                f"Executing stop loss for {position_id}: {reason}"
            )
            
            # Publish stop loss triggered event
            await self.event_bus.publish(
                StopLossEvents.STOP_LOSS_TRIGGERED,
                {
                    "stop_loss_type": stop_loss_type,
                    "position_id": position_id,
                    "position": position,
                    "reason": reason,
                    "timestamp": time.time()
                }
            )
            
            # Execute the stop loss
            if partial_close_enabled:
                # Close part of the position
                close_percentage = partial_close_percentage * 100  # Convert to percentage
                result = await self.position_manager.close_position(
                    side=position_side,
                    percentage=close_percentage,
                    order_type=order_type
                )
                
                self.logger.info(
                    f"Partially closed position {position_id} by {close_percentage:.1f}% due to stop loss: {reason}"
                )
            else:
                # Close the entire position
                result = await self.position_manager.close_position(
                    side=position_side,
                    order_type=order_type
                )
                
                self.logger.info(
                    f"Fully closed position {position_id} due to stop loss: {reason}"
                )
            
            # Update stop loss metrics
            self.stop_loss_metrics["stop_loss_triggers_count"] += 1
            self.stop_loss_metrics["last_stop_loss_time"] = time.time()
            
            # Add to stop loss history
            self.stop_loss_history.append({
                "position_id": position_id,
                "stop_loss_type": stop_loss_type,
                "reason": reason,
                "timestamp": time.time(),
                "result": result
            })
            
            # Publish stop loss executed event
            await self.event_bus.publish(
                StopLossEvents.STOP_LOSS_EXECUTED,
                {
                    "stop_loss_type": stop_loss_type,
                    "position_id": position_id,
                    "position": position,
                    "reason": reason,
                    "result": result,
                    "timestamp": time.time()
                }
            )
        except Exception as e:
            self.logger.error(f"Error executing stop loss: {e}")
    
    async def _handle_position_opened(self, event_data: Dict[str, Any]) -> None:
        """
        Handle position opened events.
        
        Args:
            event_data: Position opened event data
        """
        try:
            position = event_data.get('position', {})
            position_id = position.get('position_id', f"{position.get('pair')}-{position.get('position_side')}")
            
            # Add position to stop loss tracking
            self.stop_loss_metrics["positions_with_stop_loss"][position_id] = {
                "entry_price": position.get('entry_price', 0),
                "size": position.get('size', 0),
                "initial_value": position.get('size', 0) * position.get('entry_price', 0),
                "current_value": position.get('size', 0) * position.get('entry_price', 0),
                "max_value": position.get('size', 0) * position.get('entry_price', 0),
                "unrealized_pnl": position.get('unrealized_pnl', 0),
                "stop_loss_price": 0.0,
                "stop_loss_triggered": False
            }
            
            self.logger.info(f"Added position {position_id} to stop loss tracking")
        except Exception as e:
            self.logger.error(f"Error handling position opened event: {e}")
    
    async def _handle_position_modified(self, event_data: Dict[str, Any]) -> None:
        """
        Handle position modified events.
        
        Args:
            event_data: Position modified event data
        """
        try:
            position = event_data.get('position', {})
            position_id = position.get('position_id', f"{position.get('pair')}-{position.get('position_side')}")
            
            # Update position in stop loss tracking
            if position_id in self.stop_loss_metrics["positions_with_stop_loss"]:
                self.stop_loss_metrics["positions_with_stop_loss"][position_id].update({
                    "size": position.get('size', 0),
                    "current_value": position.get('size', 0) * position.get('entry_price', 0),
                    "unrealized_pnl": position.get('unrealized_pnl', 0)
                })
                
                self.logger.debug(f"Updated position {position_id} in stop loss tracking")
        except Exception as e:
            self.logger.error(f"Error handling position modified event: {e}")
    
    async def _handle_position_closed(self, event_data: Dict[str, Any]) -> None:
        """
        Handle position closed events.
        
        Args:
            event_data: Position closed event data
        """
        try:
            position = event_data.get('position', {})
            position_id = position.get('position_id', f"{position.get('pair')}-{position.get('position_side')}")
            
            # Remove position from stop loss tracking
            if position_id in self.stop_loss_metrics["positions_with_stop_loss"]:
                del self.stop_loss_metrics["positions_with_stop_loss"][position_id]
                
                self.logger.info(f"Removed position {position_id} from stop loss tracking")
        except Exception as e:
            self.logger.error(f"Error handling position closed event: {e}")
    
    async def _handle_position_pnl_update(self, event_data: Dict[str, Any]) -> None:
        """
        Handle position PnL update events.
        
        Args:
            event_data: Position PnL update event data
        """
        try:
            position = event_data.get('position', {})
            position_id = position.get('position_id', f"{position.get('pair')}-{position.get('position_side')}")
            unrealized_pnl = event_data.get('unrealized_pnl', 0)
            current_price = event_data.get('current_price', 0)
            
            # Update position in stop loss tracking
            if position_id in self.stop_loss_metrics["positions_with_stop_loss"]:
                position_data = self.stop_loss_metrics["positions_with_stop_loss"][position_id]
                position_data["unrealized_pnl"] = unrealized_pnl
                
                # Update current value
                size = position_data["size"]
                position_data["current_value"] = size * current_price
                
                # Update max value if current value is higher
                if position_data["current_value"] > position_data["max_value"]:
                    position_data["max_value"] = position_data["current_value"]
        except Exception as e:
            self.logger.error(f"Error handling position PnL update event: {e}")
    
    async def _handle_liquidation_risk(self, event_data: Dict[str, Any]) -> None:
        """
        Handle liquidation risk events from the risk manager.
        
        Args:
            event_data: Liquidation risk event data
        """
        try:
            # If liquidation risk is high, consider executing stop loss
            risk_factor = event_data.get('risk_factor', 0.0)
            position = event_data.get('position', {})
            
            # If risk factor is very high (e.g., > 0.9), execute stop loss
            if risk_factor > 0.9:
                await self._execute_stop_loss(
                    position=position,
                    reason=f"Liquidation risk too high: risk factor {risk_factor:.2f}",
                    stop_loss_type="liquidation_protection"
                )
        except Exception as e:
            self.logger.error(f"Error handling liquidation risk event: {e}")
    
    async def _handle_circuit_breaker(self, event_data: Dict[str, Any]) -> None:
        """
        Handle circuit breaker events from the risk manager.
        
        Args:
            event_data: Circuit breaker event data
        """
        try:
            # Circuit breaker triggered, consider closing positions
            # This is a placeholder for potential integration with circuit breaker events
            pass
        except Exception as e:
            self.logger.error(f"Error handling circuit breaker event: {e}")
    
    async def _handle_drawdown_exceeded(self, event_data: Dict[str, Any]) -> None:
        """
        Handle drawdown exceeded events from the risk manager.
        
        Args:
            event_data: Drawdown exceeded event data
        """
        try:
            # If drawdown is critical, consider executing stop loss
            alert_level = event_data.get('alert_level', '')
            current_drawdown = event_data.get('current_drawdown', 0.0)
            
            # If alert level is critical, execute stop loss
            if alert_level == 'critical' and current_drawdown > 0.25:  # 25% drawdown
                # Get all positions
                positions = await self.position_manager.get_all_positions()
                
                # Execute stop loss for all positions
                for position in positions:
                    await self._execute_stop_loss(
                        position=position,
                        reason=f"Critical drawdown: {current_drawdown:.2%}",
                        stop_loss_type="drawdown_protection"
                    )
        except Exception as e:
            self.logger.error(f"Error handling drawdown exceeded event: {e}")
    
    async def update_stop_loss_settings(self, new_settings: Dict[str, Any]) -> None:
        """
        Update stop loss settings.
        
        Args:
            new_settings: New stop loss settings
        """
        try:
            # Update settings
            for category, settings in new_settings.items():
                if category in self.stop_loss_settings:
                    self.stop_loss_settings[category].update(settings)
            
            # Publish settings updated event
            await self.event_bus.publish(
                StopLossEvents.STOP_LOSS_SETTINGS_UPDATED,
                {
                    "old_settings": self.stop_loss_settings.copy(),
                    "new_settings": self.stop_loss_settings,
                    "timestamp": time.time()
                }
            )
            
            self.logger.info(f"Updated stop loss settings: {new_settings}")
        except Exception as e:
            self.logger.error(f"Error updating stop loss settings: {e}")
    
    async def get_stop_loss_metrics(self) -> Dict[str, Any]:
        """
        Get current stop loss metrics.
        
        Returns:
            Current stop loss metrics
        """
        return {
            "metrics": self.stop_loss_metrics,
            "settings": self.stop_loss_settings,
            "history": self.stop_loss_history,
            "timestamp": time.time()
        }
    
    async def shutdown(self) -> None:
        """
        Shutdown the stop loss manager, cancelling all background tasks.
        """
        self.logger.info("Shutting down StopLossManager...")
        
        # Cancel all stop loss update tasks
        for task in self.stop_loss_update_tasks:
            task.cancel()
        
        # Wait for all tasks to complete
        if self.stop_loss_update_tasks:
            await asyncio.gather(*self.stop_loss_update_tasks, return_exceptions=True)
        
        self.logger.info("StopLossManager shutdown complete")
