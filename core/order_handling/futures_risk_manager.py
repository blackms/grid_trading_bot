import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Union
from decimal import Decimal
import time

from config.config_manager import ConfigManager
from core.services.exchange_interface import ExchangeInterface
from core.bot_management.event_bus import EventBus, Events
from core.order_handling.futures_position_manager import FuturesPositionManager, Position, PositionEvents

class RiskEvents:
    """
    Defines risk-related event types for the EventBus.
    """
    LIQUIDATION_RISK_DETECTED = "liquidation_risk_detected"
    CIRCUIT_BREAKER_TRIGGERED = "circuit_breaker_triggered"
    MARGIN_HEALTH_WARNING = "margin_health_warning"
    DRAWDOWN_THRESHOLD_EXCEEDED = "drawdown_threshold_exceeded"
    POSITION_SIZE_LIMIT_REACHED = "position_size_limit_reached"
    FUNDING_RATE_ALERT = "funding_rate_alert"
    RISK_METRICS_UPDATE = "risk_metrics_update"

class FuturesRiskManager:
    """
    Manages risk for futures trading operations.
    
    Responsibilities:
    1. Liquidation prevention mechanisms
    2. Position sizing algorithms for leveraged trading
    3. Circuit breakers and safety mechanisms
    4. Margin health monitoring
    5. Funding rate impact estimation
    6. Drawdown management
    7. Risk metrics tracking and reporting
    """
    
    def __init__(
        self,
        config_manager: ConfigManager,
        exchange_service: ExchangeInterface,
        event_bus: EventBus,
        position_manager: FuturesPositionManager
    ):
        """
        Initialize the FuturesRiskManager.
        
        Args:
            config_manager: Configuration manager instance
            exchange_service: Exchange service instance
            event_bus: Event bus for publishing risk events
            position_manager: Futures position manager instance
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config_manager = config_manager
        self.exchange_service = exchange_service
        self.event_bus = event_bus
        self.position_manager = position_manager
        
        # Validate that we're in futures mode
        if not self.config_manager.is_futures_market():
            raise ValueError("FuturesRiskManager can only be used with futures markets")
        
        # Initialize risk tracking
        self.risk_metrics = {}
        self.risk_update_tasks = set()
        self.circuit_breaker_active = False
        self.last_funding_rates = {}
        self.drawdown_tracking = {}
        self.risk_limits = {}
        
        # Get trading pair
        self.base_currency = self.config_manager.get_base_currency()
        self.quote_currency = self.config_manager.get_quote_currency()
        self.pair = f"{self.base_currency}/{self.quote_currency}"
        self.exchange_pair = self.pair.replace("/", "")
        
        # Get futures settings
        self.leverage = self.config_manager.get_leverage()
        self.margin_type = self.config_manager.get_margin_type()
        
        # Load risk management settings
        self._load_risk_settings()
        
        self.logger.info(f"Initialized FuturesRiskManager for {self.pair} with leverage {self.leverage}x")
    
    def _load_risk_settings(self) -> None:
        """
        Load risk management settings from configuration.
        """
        # Liquidation protection settings
        futures_risk = self.config_manager.get_futures_risk_management()
        liquidation_protection = self.config_manager.get_liquidation_protection()
        
        self.liquidation_protection_enabled = self.config_manager.is_liquidation_protection_enabled()
        self.liquidation_protection_threshold = self.config_manager.get_liquidation_protection_threshold()
        self.max_position_size = self.config_manager.get_max_position_size()
        
        # Additional risk settings with defaults
        self.risk_limits = {
            # Circuit breaker settings
            "circuit_breaker": {
                "enabled": futures_risk.get("circuit_breaker", {}).get("enabled", True),
                "price_change_threshold": futures_risk.get("circuit_breaker", {}).get("price_change_threshold", 0.1),  # 10% price change
                "volume_spike_threshold": futures_risk.get("circuit_breaker", {}).get("volume_spike_threshold", 3.0),  # 3x normal volume
                "cooldown_period": futures_risk.get("circuit_breaker", {}).get("cooldown_period", 300),  # 5 minutes
                "max_daily_triggers": futures_risk.get("circuit_breaker", {}).get("max_daily_triggers", 3)
            },
            
            # Margin health settings
            "margin_health": {
                "warning_threshold": futures_risk.get("margin_health", {}).get("warning_threshold", 0.5),  # 50% of maintenance margin
                "critical_threshold": futures_risk.get("margin_health", {}).get("critical_threshold", 0.2),  # 20% of maintenance margin
                "auto_reduce_enabled": futures_risk.get("margin_health", {}).get("auto_reduce_enabled", False),
                "auto_reduce_percentage": futures_risk.get("margin_health", {}).get("auto_reduce_percentage", 0.25)  # Reduce by 25%
            },
            
            # Funding rate settings
            "funding_rate": {
                "high_threshold": futures_risk.get("funding_rate", {}).get("high_threshold", 0.001),  # 0.1% per 8 hours
                "extreme_threshold": futures_risk.get("funding_rate", {}).get("extreme_threshold", 0.003),  # 0.3% per 8 hours
                "cumulative_threshold": futures_risk.get("funding_rate", {}).get("cumulative_threshold", 0.01),  # 1% cumulative
                "auto_hedge_enabled": futures_risk.get("funding_rate", {}).get("auto_hedge_enabled", False)
            },
            
            # Drawdown management
            "drawdown": {
                "warning_threshold": futures_risk.get("drawdown", {}).get("warning_threshold", 0.1),  # 10% drawdown
                "critical_threshold": futures_risk.get("drawdown", {}).get("critical_threshold", 0.2),  # 20% drawdown
                "max_drawdown": futures_risk.get("drawdown", {}).get("max_drawdown", 0.3),  # 30% max drawdown
                "auto_close_enabled": futures_risk.get("drawdown", {}).get("auto_close_enabled", False)
            },
            
            # Position sizing
            "position_sizing": {
                "max_capital_allocation": futures_risk.get("position_sizing", {}).get("max_capital_allocation", 0.2),  # 20% of capital
                "scale_with_volatility": futures_risk.get("position_sizing", {}).get("scale_with_volatility", True),
                "max_notional_value": futures_risk.get("position_sizing", {}).get("max_notional_value", None),
                "dynamic_leverage": futures_risk.get("position_sizing", {}).get("dynamic_leverage", True)
            }
        }
        
        self.logger.info(f"Loaded risk management settings: liquidation protection: {self.liquidation_protection_enabled}, threshold: {self.liquidation_protection_threshold}")
    
    async def initialize(self) -> None:
        """
        Initialize the risk manager and start monitoring tasks.
        """
        try:
            # Initialize risk metrics
            await self._initialize_risk_metrics()
            
            # Start risk monitoring tasks
            self._start_risk_monitoring()
            
            # Subscribe to position events
            await self._subscribe_to_position_events()
            
            self.logger.info(f"FuturesRiskManager initialized successfully for {self.pair}")
        except Exception as e:
            self.logger.error(f"Error initializing FuturesRiskManager: {e}")
            raise
    
    async def _initialize_risk_metrics(self) -> None:
        """
        Initialize risk metrics with current market data.
        """
        try:
            # Get current price
            current_price = await self.exchange_service.get_current_price(self.exchange_pair)
            
            # Get current funding rate
            funding_rate_data = await self.exchange_service.get_funding_rate(self.exchange_pair)
            current_funding_rate = funding_rate_data.get('fundingRate', 0.0)
            next_funding_time = funding_rate_data.get('nextFundingTime', 0)
            
            # Get contract specifications
            contract_specs = await self.exchange_service.get_contract_specifications(self.exchange_pair)
            
            # Initialize risk metrics
            self.risk_metrics = {
                "last_price": current_price,
                "price_24h_high": current_price,
                "price_24h_low": current_price,
                "price_change_24h": 0.0,
                "current_funding_rate": current_funding_rate,
                "next_funding_time": next_funding_time,
                "funding_payments_24h": 0.0,
                "liquidation_risk_level": 0.0,  # 0.0 to 1.0, where 1.0 is imminent liquidation
                "margin_health": 1.0,  # 1.0 is healthy, 0.0 is critical
                "max_drawdown": 0.0,
                "current_drawdown": 0.0,
                "circuit_breaker_triggers_24h": 0,
                "last_circuit_breaker_time": 0,
                "position_size_utilization": 0.0,  # 0.0 to 1.0, where 1.0 is max position size
                "risk_adjusted_leverage": self.leverage,
                "max_safe_position_size": 0.0,
                "contract_specs": contract_specs
            }
            
            # Initialize last funding rates
            self.last_funding_rates = {
                "current": current_funding_rate,
                "history": [current_funding_rate],
                "cumulative_24h": 0.0
            }
            
            # Initialize drawdown tracking
            self.drawdown_tracking = {
                "peak_equity": 0.0,
                "current_equity": 0.0,
                "max_drawdown": 0.0,
                "drawdown_start_time": 0,
                "in_drawdown": False
            }
            
            self.logger.info(f"Initialized risk metrics with current price: {current_price}, funding rate: {current_funding_rate}")
        except Exception as e:
            self.logger.error(f"Error initializing risk metrics: {e}")
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
            
            # Subscribe to position liquidation warning events
            await self.event_bus.subscribe(PositionEvents.POSITION_LIQUIDATION_WARNING, self._handle_position_liquidation_warning)
            
            self.logger.info("Subscribed to position events")
        except Exception as e:
            self.logger.error(f"Error subscribing to position events: {e}")
            raise
    
    def _start_risk_monitoring(self) -> None:
        """
        Start background tasks for monitoring various risk factors.
        """
        # Start liquidation risk monitoring
        liquidation_task = asyncio.create_task(self._monitor_liquidation_risk())
        self.risk_update_tasks.add(liquidation_task)
        liquidation_task.add_done_callback(self.risk_update_tasks.discard)
        
        # Start circuit breaker monitoring
        circuit_breaker_task = asyncio.create_task(self._monitor_circuit_breakers())
        self.risk_update_tasks.add(circuit_breaker_task)
        circuit_breaker_task.add_done_callback(self.risk_update_tasks.discard)
        
        # Start margin health monitoring
        margin_task = asyncio.create_task(self._monitor_margin_health())
        self.risk_update_tasks.add(margin_task)
        margin_task.add_done_callback(self.risk_update_tasks.discard)
        
        # Start funding rate monitoring
        funding_task = asyncio.create_task(self._monitor_funding_rates())
        self.risk_update_tasks.add(funding_task)
        funding_task.add_done_callback(self.risk_update_tasks.discard)
        
        # Start drawdown monitoring
        drawdown_task = asyncio.create_task(self._monitor_drawdown())
        self.risk_update_tasks.add(drawdown_task)
        drawdown_task.add_done_callback(self.risk_update_tasks.discard)
        
        # Start risk metrics reporting
        metrics_task = asyncio.create_task(self._report_risk_metrics())
        self.risk_update_tasks.add(metrics_task)
        metrics_task.add_done_callback(self.risk_update_tasks.discard)
        
        self.logger.info("Started risk monitoring tasks")
    
    async def _monitor_liquidation_risk(self) -> None:
        """
        Monitor positions for liquidation risk and take preventive actions if necessary.
        """
        try:
            while True:
                # Skip if liquidation protection is disabled
                if not self.liquidation_protection_enabled:
                    await asyncio.sleep(30)
                    continue
                
                # Get all positions
                positions = await self.position_manager.get_all_positions()
                
                for position_data in positions:
                    position_side = position_data.get('position_side')
                    liquidation_price = position_data.get('liquidation_price')
                    
                    if not liquidation_price:
                        continue
                    
                    # Get current price
                    current_price = await self.exchange_service.get_current_price(self.exchange_pair)
                    
                    # Calculate distance to liquidation
                    if position_side == 'long':
                        distance_to_liquidation = (current_price - liquidation_price) / current_price
                    else:
                        distance_to_liquidation = (liquidation_price - current_price) / current_price
                    
                    # Update risk metrics
                    self.risk_metrics["liquidation_risk_level"] = max(0.0, 1.0 - (distance_to_liquidation / self.liquidation_protection_threshold))
                    
                    # Check if we're below the threshold
                    if distance_to_liquidation < self.liquidation_protection_threshold:
                        # Calculate how much of the position to reduce to move away from liquidation
                        # The closer to liquidation, the more we reduce
                        risk_factor = 1.0 - (distance_to_liquidation / self.liquidation_protection_threshold)
                        reduction_percentage = min(0.5, risk_factor)  # Max 50% reduction
                        
                        # Log the liquidation risk
                        self.logger.warning(
                            f"Liquidation risk detected for {position_data.get('pair')} {position_side}: "
                            f"Current price: {current_price}, Liquidation price: {liquidation_price}, "
                            f"Distance: {distance_to_liquidation:.2%}, Risk factor: {risk_factor:.2f}"
                        )
                        
                        # Publish liquidation risk event
                        await self.event_bus.publish(
                            RiskEvents.LIQUIDATION_RISK_DETECTED,
                            {
                                "pair": position_data.get('pair'),
                                "position_side": position_side,
                                "current_price": current_price,
                                "liquidation_price": liquidation_price,
                                "distance_to_liquidation": distance_to_liquidation,
                                "threshold": self.liquidation_protection_threshold,
                                "risk_factor": risk_factor,
                                "recommended_reduction": reduction_percentage,
                                "position": position_data
                            }
                        )
                        
                        # If auto-reduce is enabled in margin health settings, reduce the position
                        if self.risk_limits["margin_health"]["auto_reduce_enabled"] and risk_factor > 0.7:  # Only auto-reduce if risk is high
                            try:
                                # Calculate reduction amount
                                position_size = position_data.get('size', 0)
                                reduction_amount = position_size * reduction_percentage
                                
                                self.logger.warning(f"Auto-reducing position by {reduction_percentage:.2%} ({reduction_amount} {self.base_currency}) to prevent liquidation")
                                
                                # Close part of the position
                                await self.position_manager.close_position(
                                    side=position_side,
                                    percentage=reduction_percentage * 100,  # Convert to percentage
                                    order_type="market"  # Use market order for immediate execution
                                )
                            except Exception as e:
                                self.logger.error(f"Failed to auto-reduce position: {e}")
                
                # Check every 15 seconds
                await asyncio.sleep(15)
        except asyncio.CancelledError:
            self.logger.info("Liquidation risk monitoring task cancelled")
        except Exception as e:
            self.logger.error(f"Error in liquidation risk monitoring: {e}")
    
    async def _monitor_circuit_breakers(self) -> None:
        """
        Monitor market conditions for circuit breaker triggers.
        """
        try:
            # Initialize price history for volatility calculation
            price_history = []
            volume_history = []
            last_check_time = time.time()
            circuit_breaker_settings = self.risk_limits["circuit_breaker"]
            
            while True:
                # Skip if circuit breaker is disabled
                if not circuit_breaker_settings["enabled"]:
                    await asyncio.sleep(30)
                    continue
                
                current_time = time.time()
                
                # Get current price and 24h OHLCV data
                current_price = await self.exchange_service.get_current_price(self.exchange_pair)
                
                # Add to price history (keep last 24 data points for hourly checks)
                price_history.append(current_price)
                if len(price_history) > 24:
                    price_history.pop(0)
                
                # Check if we have enough data
                if len(price_history) < 2:
                    await asyncio.sleep(60)
                    continue
                
                # Calculate price change since last check
                price_change = abs(current_price - price_history[-2]) / price_history[-2]
                
                # Update 24h high/low
                self.risk_metrics["price_24h_high"] = max(self.risk_metrics["price_24h_high"], current_price)
                self.risk_metrics["price_24h_low"] = min(self.risk_metrics["price_24h_low"], current_price)
                
                # Calculate 24h price change
                if len(price_history) >= 24:
                    self.risk_metrics["price_change_24h"] = (current_price - price_history[0]) / price_history[0]
                
                # Check for circuit breaker conditions
                if (price_change > circuit_breaker_settings["price_change_threshold"] and
                    self.risk_metrics["circuit_breaker_triggers_24h"] < circuit_breaker_settings["max_daily_triggers"] and
                    (current_time - self.risk_metrics["last_circuit_breaker_time"]) > circuit_breaker_settings["cooldown_period"]):
                    
                    # Trigger circuit breaker
                    self.circuit_breaker_active = True
                    self.risk_metrics["circuit_breaker_triggers_24h"] += 1
                    self.risk_metrics["last_circuit_breaker_time"] = current_time
                    
                    self.logger.warning(
                        f"Circuit breaker triggered: Price change of {price_change:.2%} exceeds threshold of "
                        f"{circuit_breaker_settings['price_change_threshold']:.2%}"
                    )
                    
                    # Publish circuit breaker event
                    await self.event_bus.publish(
                        RiskEvents.CIRCUIT_BREAKER_TRIGGERED,
                        {
                            "pair": self.pair,
                            "current_price": current_price,
                            "price_change": price_change,
                            "threshold": circuit_breaker_settings["price_change_threshold"],
                            "cooldown_period": circuit_breaker_settings["cooldown_period"],
                            "triggers_24h": self.risk_metrics["circuit_breaker_triggers_24h"],
                            "max_triggers": circuit_breaker_settings["max_daily_triggers"]
                        }
                    )
                    
                    # Reset after cooldown period
                    await asyncio.sleep(circuit_breaker_settings["cooldown_period"])
                    self.circuit_breaker_active = False
                    self.logger.info(f"Circuit breaker reset after cooldown period of {circuit_breaker_settings['cooldown_period']} seconds")
                
                # Reset daily trigger count if 24 hours have passed
                if (current_time - last_check_time) > 86400:  # 24 hours in seconds
                    self.risk_metrics["circuit_breaker_triggers_24h"] = 0
                    last_check_time = current_time
                
                # Check every minute
                await asyncio.sleep(60)
        except asyncio.CancelledError:
            self.logger.info("Circuit breaker monitoring task cancelled")
        except Exception as e:
            self.logger.error(f"Error in circuit breaker monitoring: {e}")
    
    async def _monitor_margin_health(self) -> None:
        """
        Monitor margin health and take actions if it falls below thresholds.
        """
        try:
            margin_health_settings = self.risk_limits["margin_health"]
            
            while True:
                # Get all positions
                positions = await self.position_manager.get_all_positions()
                
                if not positions:
                    # No open positions, margin health is perfect
                    self.risk_metrics["margin_health"] = 1.0
                    await asyncio.sleep(30)
                    continue
                
                # Get account balance
                balance_data = await self.exchange_service.get_balance()
                available_margin = float(balance_data.get(self.quote_currency, {}).get('free', 0))
                used_margin = float(balance_data.get(self.quote_currency, {}).get('used', 0))
                total_margin = available_margin + used_margin
                
                # Calculate margin health as ratio of available to used margin
                if used_margin > 0:
                    margin_ratio = available_margin / used_margin
                    
                    # Update risk metrics
                    self.risk_metrics["margin_health"] = min(1.0, margin_ratio)
                    
                    # Check if margin health is below warning threshold
                    if margin_ratio < margin_health_settings["warning_threshold"]:
                        self.logger.warning(
                            f"Margin health warning: Available margin ({available_margin:.2f} {self.quote_currency}) "
                            f"is {margin_ratio:.2f}x used margin ({used_margin:.2f} {self.quote_currency})"
                        )
                        
                        # Publish margin health warning event
                        await self.event_bus.publish(
                            RiskEvents.MARGIN_HEALTH_WARNING,
                            {
                                "available_margin": available_margin,
                                "used_margin": used_margin,
                                "total_margin": total_margin,
                                "margin_ratio": margin_ratio,
                                "warning_threshold": margin_health_settings["warning_threshold"],
                                "critical_threshold": margin_health_settings["critical_threshold"]
                            }
                        )
                        
                        # If margin health is below critical threshold and auto-reduce is enabled, reduce positions
                        if margin_ratio < margin_health_settings["critical_threshold"] and margin_health_settings["auto_reduce_enabled"]:
                            self.logger.warning(f"Critical margin health: Auto-reducing positions by {margin_health_settings['auto_reduce_percentage']:.2%}")
                            
                            # Reduce each position by the configured percentage
                            for position_data in positions:
                                try:
                                    position_side = position_data.get('position_side')
                                    
                                    # Close part of the position
                                    await self.position_manager.close_position(
                                        side=position_side,
                                        percentage=margin_health_settings["auto_reduce_percentage"] * 100,  # Convert to percentage
                                        order_type="market"  # Use market order for immediate execution
                                    )
                                except Exception as e:
                                    self.logger.error(f"Failed to auto-reduce position: {e}")
                
                # Check every 30 seconds
                await asyncio.sleep(30)
        except asyncio.CancelledError:
            self.logger.info("Margin health monitoring task cancelled")
        except Exception as e:
            self.logger.error(f"Error in margin health monitoring: {e}")
    
    async def _monitor_funding_rates(self) -> None:
        """
        Monitor funding rates and estimate their impact on positions.
        """
        try:
            funding_rate_settings = self.risk_limits["funding_rate"]
            
            while True:
                # Get current funding rate
                funding_rate_data = await self.exchange_service.get_funding_rate(self.exchange_pair)
                current_funding_rate = funding_rate_data.get('fundingRate', 0.0)
                next_funding_time = funding_rate_data.get('nextFundingTime', 0)
                
                # Update funding rate history
                self.last_funding_rates["current"] = current_funding_rate
                self.last_funding_rates["history"].append(current_funding_rate)
                
                # Keep only the last 24 funding rates (8 hours * 3 = 24 hours)
                if len(self.last_funding_rates["history"]) > 3:
                    self.last_funding_rates["history"] = self.last_funding_rates["history"][-3:]
                
                # Calculate cumulative funding rate for the last 24 hours
                self.last_funding_rates["cumulative_24h"] = sum(self.last_funding_rates["history"])
                
                # Update risk metrics
                self.risk_metrics["current_funding_rate"] = current_funding_rate
                self.risk_metrics["next_funding_time"] = next_funding_time
                
                # Get all positions
                positions = await self.position_manager.get_all_positions()
                
                # Calculate funding impact on each position
                total_funding_impact = 0.0
                for position_data in positions:
                    position_side = position_data.get('position_side')
                    position_size = position_data.get('size', 0)
                    entry_price = position_data.get('entry_price', 0)
                    
                    # Calculate position value
                    position_value = position_size * entry_price
                    
                    # Calculate funding impact (positive for payment received, negative for payment made)
                    # For long positions: negative funding rate means payment received
                    # For short positions: positive funding rate means payment received
                    funding_impact = position_value * current_funding_rate
                    if position_side == 'long':
                        funding_impact = -funding_impact
                    
                    total_funding_impact += funding_impact
                
                # Update funding payments in risk metrics
                self.risk_metrics["funding_payments_24h"] = total_funding_impact
                
                # Check if funding rate exceeds thresholds
                abs_funding_rate = abs(current_funding_rate)
                if abs_funding_rate > funding_rate_settings["extreme_threshold"]:
                    self.logger.warning(
                        f"Extreme funding rate detected: {current_funding_rate:.6f} "
                        f"(threshold: {funding_rate_settings['extreme_threshold']:.6f})"
                    )
                    
                    # Publish funding rate alert event
                    await self.event_bus.publish(
                        RiskEvents.FUNDING_RATE_ALERT,
                        {
                            "pair": self.pair,
                            "current_funding_rate": current_funding_rate,
                            "threshold": funding_rate_settings["extreme_threshold"],
                            "next_funding_time": next_funding_time,
                            "estimated_impact": total_funding_impact,
                            "alert_level": "extreme"
                        }
                    )
                elif abs_funding_rate > funding_rate_settings["high_threshold"]:
                    self.logger.info(
                        f"High funding rate detected: {current_funding_rate:.6f} "
                        f"(threshold: {funding_rate_settings['high_threshold']:.6f})"
                    )
                    
                    # Publish funding rate alert event
                    await self.event_bus.publish(
                        RiskEvents.FUNDING_RATE_ALERT,
                        {
                            "pair": self.pair,
                            "current_funding_rate": current_funding_rate,
                            "threshold": funding_rate_settings["high_threshold"],
                            "next_funding_time": next_funding_time,
                            "estimated_impact": total_funding_impact,
                            "alert_level": "high"
                        }
                    )
                
                # Check if cumulative funding rate exceeds threshold
                abs_cumulative_rate = abs(self.last_funding_rates["cumulative_24h"])
                if abs_cumulative_rate > funding_rate_settings["cumulative_threshold"]:
                    self.logger.warning(
                        f"Cumulative funding rate exceeds threshold: {self.last_funding_rates['cumulative_24h']:.6f} "
                        f"(threshold: {funding_rate_settings['cumulative_threshold']:.6f})"
                    )
                    
                    # Publish funding rate alert event
                    await self.event_bus.publish(
                        RiskEvents.FUNDING_RATE_ALERT,
                        {
                            "pair": self.pair,
                            "cumulative_funding_rate": self.last_funding_rates["cumulative_24h"],
                            "threshold": funding_rate_settings["cumulative_threshold"],
                            "estimated_impact": total_funding_impact * 3,  # Estimate for 24 hours
                            "alert_level": "cumulative"
                        }
                    )
                
                # Check every 10 minutes (funding rates typically update every 8 hours)
                await asyncio.sleep(600)
        except asyncio.CancelledError:
            self.logger.info("Funding rate monitoring task cancelled")
        except Exception as e:
            self.logger.error(f"Error in funding rate monitoring: {e}")
    
    async def _monitor_drawdown(self) -> None:
        """
        Monitor account drawdown and take actions if it exceeds thresholds.
        """
        try:
            drawdown_settings = self.risk_limits["drawdown"]
            
            while True:
                # Get account balance
                balance_data = await self.exchange_service.get_balance()
                total_equity = float(balance_data.get(self.quote_currency, {}).get('total', 0))
                
                # Get unrealized PnL from positions
                positions = await self.position_manager.get_all_positions()
                unrealized_pnl = sum(position.get('unrealized_pnl', 0) for position in positions)
                
                # Calculate total equity including unrealized PnL
                current_equity = total_equity + unrealized_pnl
                
                # Update drawdown tracking
                if current_equity > self.drawdown_tracking["peak_equity"]:
                    # New equity peak
                    self.drawdown_tracking["peak_equity"] = current_equity
                    self.drawdown_tracking["in_drawdown"] = False
                elif self.drawdown_tracking["peak_equity"] > 0:
                    # Calculate current drawdown
                    current_drawdown = (self.drawdown_tracking["peak_equity"] - current_equity) / self.drawdown_tracking["peak_equity"]
                    
                    # Update drawdown metrics
                    self.risk_metrics["current_drawdown"] = current_drawdown
                    self.risk_metrics["max_drawdown"] = max(self.risk_metrics["max_drawdown"], current_drawdown)
                    
                    # Check if we're entering a drawdown
                    if not self.drawdown_tracking["in_drawdown"] and current_drawdown > 0:
                        self.drawdown_tracking["in_drawdown"] = True
                        self.drawdown_tracking["drawdown_start_time"] = time.time()
                    
                    # Check if drawdown exceeds warning threshold
                    if current_drawdown > drawdown_settings["warning_threshold"]:
                        self.logger.warning(
                            f"Drawdown warning: Current drawdown of {current_drawdown:.2%} exceeds warning threshold of "
                            f"{drawdown_settings['warning_threshold']:.2%}"
                        )
                        
                        # Publish drawdown warning event
                        await self.event_bus.publish(
                            RiskEvents.DRAWDOWN_THRESHOLD_EXCEEDED,
                            {
                                "current_equity": current_equity,
                                "peak_equity": self.drawdown_tracking["peak_equity"],
                                "current_drawdown": current_drawdown,
                                "warning_threshold": drawdown_settings["warning_threshold"],
                                "critical_threshold": drawdown_settings["critical_threshold"],
                                "drawdown_duration": time.time() - self.drawdown_tracking["drawdown_start_time"],
                                "alert_level": "warning"
                            }
                        )
                    
                    # Check if drawdown exceeds critical threshold
                    if current_drawdown > drawdown_settings["critical_threshold"]:
                        self.logger.warning(
                            f"Critical drawdown: Current drawdown of {current_drawdown:.2%} exceeds critical threshold of "
                            f"{drawdown_settings['critical_threshold']:.2%}"
                        )
                        
                        # Publish drawdown critical event
                        await self.event_bus.publish(
                            RiskEvents.DRAWDOWN_THRESHOLD_EXCEEDED,
                            {
                                "current_equity": current_equity,
                                "peak_equity": self.drawdown_tracking["peak_equity"],
                                "current_drawdown": current_drawdown,
                                "warning_threshold": drawdown_settings["warning_threshold"],
                                "critical_threshold": drawdown_settings["critical_threshold"],
                                "drawdown_duration": time.time() - self.drawdown_tracking["drawdown_start_time"],
                                "alert_level": "critical"
                            }
                        )
                        
                        # If auto-close is enabled and drawdown exceeds max drawdown, close all positions
                        if drawdown_settings["auto_close_enabled"] and current_drawdown > drawdown_settings["max_drawdown"]:
                            self.logger.warning(
                                f"Max drawdown exceeded: Auto-closing all positions. Drawdown: {current_drawdown:.2%}, "
                                f"Max allowed: {drawdown_settings['max_drawdown']:.2%}"
                            )
                            
                            # Close all positions
                            for position_data in positions:
                                try:
                                    position_side = position_data.get('position_side')
                                    
                                    # Close the entire position
                                    await self.position_manager.close_position(
                                        side=position_side,
                                        order_type="market"  # Use market order for immediate execution
                                    )
                                except Exception as e:
                                    self.logger.error(f"Failed to auto-close position: {e}")
                
                # Update current equity in drawdown tracking
                self.drawdown_tracking["current_equity"] = current_equity
                
                # Check every minute
                await asyncio.sleep(60)
        except asyncio.CancelledError:
            self.logger.info("Drawdown monitoring task cancelled")
        except Exception as e:
            self.logger.error(f"Error in drawdown monitoring: {e}")
    
    async def _report_risk_metrics(self) -> None:
        """
        Periodically report risk metrics to subscribers.
        """
        try:
            while True:
                # Publish risk metrics update event
                await self.event_bus.publish(
                    RiskEvents.RISK_METRICS_UPDATE,
                    {
                        "pair": self.pair,
                        "metrics": self.risk_metrics,
                        "funding_rates": self.last_funding_rates,
                        "drawdown": {
                            "current": self.risk_metrics["current_drawdown"],
                            "max": self.risk_metrics["max_drawdown"],
                            "peak_equity": self.drawdown_tracking["peak_equity"],
                            "current_equity": self.drawdown_tracking["current_equity"]
                        },
                        "circuit_breaker_active": self.circuit_breaker_active,
                        "timestamp": time.time()
                    }
                )
                
                # Report every 5 minutes
                await asyncio.sleep(300)
        except asyncio.CancelledError:
            self.logger.info("Risk metrics reporting task cancelled")
        except Exception as e:
            self.logger.error(f"Error in risk metrics reporting: {e}")
    
    # --- Position Event Handlers ---
    
    async def _handle_position_opened(self, event_data: Dict[str, Any]) -> None:
        """
        Handle position opened events.
        
        Args:
            event_data: Position opened event data
        """
        try:
            position_side = event_data.get('position_side')
            size = event_data.get('size')
            entry_price = event_data.get('entry_price')
            
            self.logger.info(f"Position opened: {position_side} {size} {self.base_currency} at {entry_price}")
            
            # Check if position size exceeds limits
            if self.max_position_size and size > self.max_position_size:
                self.logger.warning(f"Position size {size} exceeds max position size {self.max_position_size}")
                
                # Publish position size limit reached event
                await self.event_bus.publish(
                    RiskEvents.POSITION_SIZE_LIMIT_REACHED,
                    {
                        "pair": event_data.get('pair'),
                        "position_side": position_side,
                        "size": size,
                        "max_size": self.max_position_size,
                        "position": event_data.get('position')
                    }
                )
        except Exception as e:
            self.logger.error(f"Error handling position opened event: {e}")
    
    async def _handle_position_modified(self, event_data: Dict[str, Any]) -> None:
        """
        Handle position modified events.
        
        Args:
            event_data: Position modified event data
        """
        try:
            position_side = event_data.get('position_side')
            old_size = event_data.get('old_size')
            new_size = event_data.get('new_size')
            size_change = event_data.get('size_change')
            
            self.logger.info(f"Position modified: {position_side} from {old_size} to {new_size} {self.base_currency}")
            
            # Check if position size exceeds limits
            if self.max_position_size and new_size > self.max_position_size:
                self.logger.warning(f"Position size {new_size} exceeds max position size {self.max_position_size}")
                
                # Publish position size limit reached event
                await self.event_bus.publish(
                    RiskEvents.POSITION_SIZE_LIMIT_REACHED,
                    {
                        "pair": event_data.get('pair'),
                        "position_side": position_side,
                        "size": new_size,
                        "max_size": self.max_position_size,
                        "position": event_data.get('position')
                    }
                )
        except Exception as e:
            self.logger.error(f"Error handling position modified event: {e}")
    
    async def _handle_position_closed(self, event_data: Dict[str, Any]) -> None:
        """
        Handle position closed events.
        
        Args:
            event_data: Position closed event data
        """
        try:
            position_side = event_data.get('position_side')
            realized_pnl = event_data.get('realized_pnl')
            
            self.logger.info(f"Position closed: {position_side} with realized PnL {realized_pnl}")
            
            # Reset risk metrics for this position
            if position_side == 'long':
                # Reset long position risk metrics
                pass
            else:
                # Reset short position risk metrics
                pass
        except Exception as e:
            self.logger.error(f"Error handling position closed event: {e}")
    
    async def _handle_position_pnl_update(self, event_data: Dict[str, Any]) -> None:
        """
        Handle position PnL update events.
        
        Args:
            event_data: Position PnL update event data
        """
        try:
            position_side = event_data.get('position_side')
            unrealized_pnl = event_data.get('unrealized_pnl')
            current_price = event_data.get('current_price')
            
            # Update risk metrics based on PnL
            # This is handled in the drawdown monitoring task
        except Exception as e:
            self.logger.error(f"Error handling position PnL update event: {e}")
    
    async def _handle_position_liquidation_warning(self, event_data: Dict[str, Any]) -> None:
        """
        Handle position liquidation warning events.
        
        Args:
            event_data: Position liquidation warning event data
        """
        try:
            position_side = event_data.get('position_side')
            current_price = event_data.get('current_price')
            liquidation_price = event_data.get('liquidation_price')
            distance_to_liquidation = event_data.get('distance_to_liquidation')
            
            self.logger.warning(
                f"Liquidation warning from position manager: {position_side} position at risk. "
                f"Current price: {current_price}, Liquidation price: {liquidation_price}, "
                f"Distance: {distance_to_liquidation:.2%}"
            )
            
            # This is already handled in the liquidation risk monitoring task
        except Exception as e:
            self.logger.error(f"Error handling position liquidation warning event: {e}")
    
    # --- Position Sizing Methods ---
    
    async def calculate_max_position_size(self, side: str, price: float) -> float:
        """
        Calculate the maximum safe position size based on risk parameters.
        
        Args:
            side: Position side ('long' or 'short')
            price: Current price
            
        Returns:
            Maximum safe position size
        """
        try:
            position_sizing_settings = self.risk_limits["position_sizing"]
            
            # Get account balance
            balance_data = await self.exchange_service.get_balance()
            available_balance = float(balance_data.get(self.quote_currency, {}).get('free', 0))
            
            # Calculate max position size based on capital allocation
            max_capital = available_balance * position_sizing_settings["max_capital_allocation"]
            
            # Adjust for leverage
            leverage = self.leverage
            if position_sizing_settings["dynamic_leverage"]:
                # Reduce leverage for larger positions
                if max_capital > 50000:  # Example threshold
                    leverage = min(leverage, 10)  # Reduce leverage for large positions
                elif max_capital > 10000:
                    leverage = min(leverage, 20)
            
            # Calculate max position size
            max_size = (max_capital * leverage) / price
            
            # Apply max notional value limit if set
            if position_sizing_settings["max_notional_value"]:
                max_notional_size = position_sizing_settings["max_notional_value"] / price
                max_size = min(max_size, max_notional_size)
            
            # Apply max position size limit if set
            if self.max_position_size:
                max_size = min(max_size, self.max_position_size)
            
            # Scale with volatility if enabled
            if position_sizing_settings["scale_with_volatility"]:
                # Calculate volatility factor (simplified)
                volatility_factor = 1.0
                if self.risk_metrics["price_24h_high"] > 0 and self.risk_metrics["price_24h_low"] > 0:
                    price_range = (self.risk_metrics["price_24h_high"] - self.risk_metrics["price_24h_low"]) / self.risk_metrics["price_24h_low"]
                    if price_range > 0.1:  # High volatility
                        volatility_factor = 0.7  # Reduce position size
                    elif price_range < 0.03:  # Low volatility
                        volatility_factor = 1.2  # Increase position size
                
                max_size *= volatility_factor
            
            # Update risk metrics
            self.risk_metrics["max_safe_position_size"] = max_size
            self.risk_metrics["risk_adjusted_leverage"] = leverage
            
            return max_size
        except Exception as e:
            self.logger.error(f"Error calculating max position size: {e}")
            return 0.0
    
    async def is_position_within_risk_limits(self, side: str, size: float, price: float) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if a position is within risk limits.
        
        Args:
            side: Position side ('long' or 'short')
            size: Position size
            price: Position price
            
        Returns:
            Tuple of (is_within_limits, risk_info)
        """
        try:
            # Calculate max position size
            max_size = await self.calculate_max_position_size(side, price)
            
            # Check if position size exceeds max size
            is_within_limits = size <= max_size
            
            # Calculate position size utilization
            utilization = size / max_size if max_size > 0 else 1.0
            self.risk_metrics["position_size_utilization"] = utilization
            
            # Check if circuit breaker is active
            if self.circuit_breaker_active:
                is_within_limits = False
                
            # Prepare risk info
            risk_info = {
                "is_within_limits": is_within_limits,
                "max_size": max_size,
                "utilization": utilization,
                "circuit_breaker_active": self.circuit_breaker_active,
                "risk_adjusted_leverage": self.risk_metrics["risk_adjusted_leverage"]
            }
            
            return is_within_limits, risk_info
        except Exception as e:
            self.logger.error(f"Error checking position risk limits: {e}")
            return False, {"error": str(e)}
    
    async def shutdown(self) -> None:
        """
        Shutdown the risk manager, cancelling all background tasks.
        """
        self.logger.info("Shutting down FuturesRiskManager...")
        
        # Cancel all risk update tasks
        for task in self.risk_update_tasks:
            task.cancel()
        
        # Wait for all tasks to complete
        if self.risk_update_tasks:
            await asyncio.gather(*self.risk_update_tasks, return_exceptions=True)
        
        self.logger.info("FuturesRiskManager shutdown complete")
