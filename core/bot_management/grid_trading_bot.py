import logging, traceback
from typing import Optional, Dict, Any, List
from core.services.exchange_service_factory import ExchangeServiceFactory
from strategies.strategy_type import StrategyType
from strategies.grid_trading_strategy import GridTradingStrategy
from strategies.plotter import Plotter
from strategies.trading_performance_analyzer import TradingPerformanceAnalyzer
from core.order_handling.order_manager import OrderManager
from core.validation.order_validator import OrderValidator
from core.order_handling.order_status_tracker import OrderStatusTracker
from core.bot_management.event_bus import EventBus, Events
from core.order_handling.fee_calculator import FeeCalculator
from core.order_handling.balance_tracker import BalanceTracker
from core.order_handling.order_book import OrderBook
from core.grid_management.grid_manager import GridManager
from core.order_handling.execution_strategy.order_execution_strategy_factory import OrderExecutionStrategyFactory
from core.services.exceptions import UnsupportedExchangeError, DataFetchError, UnsupportedTimeframeError
from config.config_manager import ConfigManager
from config.trading_mode import TradingMode
from config.market_type import MarketType
from .notification.notification_handler import NotificationHandler
from core.order_handling.funding_rate_tracker import FundingRateTracker
from core.order_handling.futures_position_manager import FuturesPositionManager
from core.order_handling.futures_risk_manager import FuturesRiskManager
from core.order_handling.stop_loss_manager import StopLossManager

class GridTradingBot:
    def __init__(
        self, 
        config_path: str, 
        config_manager: ConfigManager,
        notification_handler: NotificationHandler,
        event_bus: EventBus,
        save_performance_results_path: Optional[str] = None, 
        no_plot: bool = False
    ):
        try:
            self.logger = logging.getLogger(self.__class__.__name__)
            self.config_path = config_path
            self.config_manager = config_manager
            self.notification_handler = notification_handler
            self.event_bus = event_bus
            self.event_bus.subscribe(Events.STOP_BOT, self._handle_stop_bot_event)
            self.event_bus.subscribe(Events.START_BOT, self._handle_start_bot_event)
            self.save_performance_results_path = save_performance_results_path
            self.no_plot = no_plot
            self.trading_mode: TradingMode = self.config_manager.get_trading_mode()
            base_currency: str = self.config_manager.get_base_currency()
            quote_currency: str = self.config_manager.get_quote_currency()
            trading_pair = f"{base_currency}/{quote_currency}"
            strategy_type: StrategyType = self.config_manager.get_strategy_type()
            market_type: MarketType = self.config_manager.get_market_type()
            self.logger.info(f"Starting Grid Trading Bot in {self.trading_mode.value} mode with strategy: {strategy_type.value}, market type: {market_type.value}")
            self.is_running = False

            self.exchange_service = ExchangeServiceFactory.create_exchange_service(self.config_manager, self.trading_mode)
            order_execution_strategy = OrderExecutionStrategyFactory.create(self.config_manager, self.exchange_service)
            grid_manager = GridManager(self.config_manager, strategy_type)
            order_validator = OrderValidator()
            fee_calculator = FeeCalculator(self.config_manager)
            
            # Initialize futures-specific components if using futures market
            self.futures_position_manager = None
            self.futures_risk_manager = None
            self.funding_rate_tracker = None
            self.dynamic_grid_manager = None
            self.stop_loss_manager = None
            
            if market_type == MarketType.FUTURES:
                self.logger.info("Initializing futures-specific components")
                self.futures_position_manager = FuturesPositionManager(
                    config_manager=self.config_manager,
                    exchange_service=self.exchange_service,
                    event_bus=self.event_bus
                )
                
                self.futures_risk_manager = FuturesRiskManager(
                    config_manager=self.config_manager,
                    exchange_service=self.exchange_service,
                    event_bus=self.event_bus,
                    position_manager=self.futures_position_manager
                )
                
                # Initialize funding rate tracker for perpetual contracts
                if self.config_manager.get_contract_type() == "perpetual":
                    self.funding_rate_tracker = FundingRateTracker(
                        config_manager=self.config_manager,
                        exchange_service=self.exchange_service,
                        event_bus=self.event_bus
                    )
                
                # Initialize dynamic grid manager for futures
                if self.config_manager.is_dynamic_grid_enabled():
                    from core.grid_management.dynamic_grid_manager import DynamicGridManager
                    self.dynamic_grid_manager = DynamicGridManager(
                        config_manager=self.config_manager,
                        strategy_type=strategy_type,
                        position_manager=self.futures_position_manager,
                        risk_manager=self.futures_risk_manager,
                        event_bus=self.event_bus
                    )
                
                # Initialize stop loss manager for futures
                self.stop_loss_manager = StopLossManager(
                    config_manager=self.config_manager,
                    exchange_service=self.exchange_service,
                    event_bus=self.event_bus,
                    position_manager=self.futures_position_manager,
                    risk_manager=self.futures_risk_manager
                )

            self.balance_tracker = BalanceTracker(
                event_bus=self.event_bus,
                fee_calculator=fee_calculator,
                trading_mode=self.trading_mode,
                base_currency=base_currency,
                quote_currency=quote_currency
            )
            order_book = OrderBook()

            self.order_status_tracker = OrderStatusTracker(
                order_book=order_book,
                order_execution_strategy=order_execution_strategy,
                event_bus=self.event_bus,
                polling_interval=5.0,
            )

            order_manager = OrderManager(
                grid_manager,
                order_validator,
                self.balance_tracker,
                order_book,
                self.event_bus,
                order_execution_strategy,
                self.notification_handler,
                self.trading_mode,
                trading_pair,
                strategy_type
            )
            
            trading_performance_analyzer = TradingPerformanceAnalyzer(self.config_manager, order_book)
            plotter = Plotter(grid_manager, order_book) if self.trading_mode == TradingMode.BACKTEST else None
            
            # Use dynamic grid manager if available, otherwise use standard grid manager
            grid_manager_to_use = self.dynamic_grid_manager if self.dynamic_grid_manager else grid_manager
            
            self.strategy = GridTradingStrategy(
                self.config_manager,
                self.event_bus,
                self.exchange_service,
                grid_manager_to_use,
                order_manager,
                self.balance_tracker,
                trading_performance_analyzer,
                self.trading_mode,
                trading_pair,
                plotter,
                self.funding_rate_tracker,
                self.futures_position_manager,
                self.futures_risk_manager,
                self.stop_loss_manager
            )

        except (UnsupportedExchangeError, DataFetchError, UnsupportedTimeframeError) as e:
            self.logger.error(f"{type(e).__name__}: {e}")
            raise

        except Exception as e:
            self.logger.error("An unexpected error occurred.")
            self.logger.error(traceback.format_exc())
            raise

    async def run(self) -> Optional[Dict[str, Any]]:
        try:
            self.is_running = True

            await self.balance_tracker.setup_balances(
                initial_balance=self.config_manager.get_initial_balance(),
                initial_crypto_balance=0.0,
                exchange_service=self.exchange_service
            )

            self.order_status_tracker.start_tracking()
            
            # Initialize futures components if applicable
            if self.config_manager.is_futures_market():
                if self.futures_position_manager:
                    await self.futures_position_manager.initialize()
                
                if self.futures_risk_manager:
                    await self.futures_risk_manager.initialize()
                
                if self.funding_rate_tracker:
                    await self.funding_rate_tracker.initialize()
                    
                if self.dynamic_grid_manager:
                    await self.dynamic_grid_manager.initialize_dynamic_grids()
                    
                if self.stop_loss_manager:
                    await self.stop_loss_manager.initialize()
            
            self.strategy.initialize_strategy()
            await self.strategy.run()

            if not self.no_plot:
                self.strategy.plot_results()

            return self._generate_and_log_performance()

        except Exception as e:
            self.logger.error(f"An unexpected error occurred {e}")
            self.logger.error(traceback.format_exc())
            raise
        
        finally:
            self.is_running = False

    async def _handle_stop_bot_event(self, reason: str) -> None:
        self.logger.info(f"Handling STOP_BOT event: {reason}")
        await self._stop()

    async def _handle_start_bot_event(self, reason: str) -> None:
        self.logger.info(f"Handling START_BOT event: {reason}")
        await self.restart()
    
    async def _stop(self) -> None:
        if not self.is_running:
            self.logger.info("Bot is not running. Nothing to stop.")
            return

        self.logger.info("Stopping Grid Trading Bot...")

        try:
            await self.order_status_tracker.stop_tracking()
            await self.strategy.stop()
            
            # Shutdown futures components if applicable
            if self.config_manager.is_futures_market():
                if self.funding_rate_tracker:
                    await self.funding_rate_tracker.shutdown()
                
                if self.stop_loss_manager:
                    await self.stop_loss_manager.shutdown()
                    
                if self.dynamic_grid_manager:
                    await self.dynamic_grid_manager.shutdown()
                
                if self.futures_risk_manager:
                    await self.futures_risk_manager.shutdown()
                
                if self.futures_position_manager:
                    await self.futures_position_manager.shutdown()
            
            self.is_running = False

        except Exception as e:
            self.logger.error(f"Error while stopping components: {e}", exc_info=True)

        self.logger.info("Grid Trading Bot has been stopped.")
    
    async def restart(self) -> None:
        if self.is_running:
            self.logger.info("Bot is already running. Restarting...")
            await self._stop()

        self.logger.info("Restarting Grid Trading Bot...")
        self.is_running = True

        try:
            self.order_status_tracker.start_tracking()
            await self.strategy.restart()

        except Exception as e:
            self.logger.error(f"Error while restarting components: {e}", exc_info=True)

        self.logger.info("Grid Trading Bot has been restarted.")

    def _generate_and_log_performance(self) -> Optional[Dict[str, Any]]:
        performance_summary, formatted_orders = self.strategy.generate_performance_report()
        return {
            "config": self.config_path,
            "performance_summary": performance_summary,
            "orders": formatted_orders
        }
    
    async def get_bot_health_status(self) -> dict:
        health_status = {
            "strategy": await self._check_strategy_health(),
            "exchange_status": await self._get_exchange_status()
        }
        
        # Add futures-specific health checks if applicable
        if self.config_manager.is_futures_market():
            if self.futures_position_manager:
                health_status["futures_position_manager"] = True  # Could add more detailed checks
            
            if self.futures_risk_manager:
                health_status["futures_risk_manager"] = True  # Could add more detailed checks
            
            if self.funding_rate_tracker:
                health_status["funding_rate_tracker"] = True  # Could add more detailed checks
                
            if self.dynamic_grid_manager:
                health_status["dynamic_grid_manager"] = True  # Could add more detailed checks
                
            if self.stop_loss_manager:
                health_status["stop_loss_manager"] = True  # Could add more detailed checks

        health_status["overall"] = all(health_status.values())
        return health_status
    
    async def _check_strategy_health(self) -> bool:
        if not self.is_running:
            self.logger.warning("Bot has stopped unexpectedly.")
            return False
        return True

    async def _get_exchange_status(self) -> str:
        exchange_status = await self.exchange_service.get_exchange_status()
        return exchange_status.get("status", "unknown")
    
    def get_balances(self) -> Dict[str, float]:
        balances = {
            "fiat": self.balance_tracker.balance,
            "reserved_fiat": self.balance_tracker.reserved_fiat,
            "crypto": self.balance_tracker.crypto_balance,
            "reserved_crypto": self.balance_tracker.reserved_crypto
        }
        
        # Add futures-specific balance information if applicable
        if self.config_manager.is_futures_market() and self.futures_position_manager:
            balances["futures"] = {
                "positions": self.futures_position_manager.get_all_positions()
            }
            
            # Add funding rate information if available
            if self.funding_rate_tracker:
                balances["futures"]["current_funding_rate"] = self.funding_rate_tracker.current_funding_rate
                if self.funding_rate_tracker.next_funding_time:
                    balances["futures"]["next_funding_time"] = self.funding_rate_tracker.next_funding_time.isoformat()
            
            # Add risk metrics if available
            if self.futures_risk_manager:
                balances["futures"]["risk_metrics"] = self.futures_risk_manager.risk_metrics
                
            # Add stop loss metrics if available
            if self.stop_loss_manager:
                balances["futures"]["stop_loss_metrics"] = self.stop_loss_manager.stop_loss_metrics
        
        return balances
    
    async def get_funding_rate_info(self) -> Optional[Dict[str, Any]]:
        """
        Get current funding rate information if using perpetual futures.
        
        Returns:
            Dictionary with funding rate information or None if not applicable
        """
        if not self.config_manager.is_futures_market() or not self.funding_rate_tracker:
            return None
            
        return await self.funding_rate_tracker.get_current_funding_info()
    
    async def get_funding_rate_forecast(self) -> Optional[List[Dict[str, Any]]]:
        """
        Get funding rate forecast if using perpetual futures.
        
        Returns:
            List of forecasted funding rates or None if not applicable
        """
        if not self.config_manager.is_futures_market() or not self.funding_rate_tracker:
            return None
            
        return await self.funding_rate_tracker.forecast_funding_rates()
        
    async def get_stop_loss_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Get stop loss metrics if using futures.
        
        Returns:
            Dictionary with stop loss metrics or None if not applicable
        """
        if not self.config_manager.is_futures_market() or not self.stop_loss_manager:
            return None
            
        return await self.stop_loss_manager.get_stop_loss_metrics()
        
    async def update_stop_loss_settings(self, new_settings: Dict[str, Any]) -> bool:
        """
        Update stop loss settings if using futures.
        
        Args:
            new_settings: Dictionary with new stop loss settings
            
        Returns:
            True if settings were updated successfully, False otherwise
        """
        if not self.config_manager.is_futures_market() or not self.stop_loss_manager:
            return False
            
        await self.stop_loss_manager.update_stop_loss_settings(new_settings)
        return True
        
    async def get_risk_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Get risk metrics if using futures.
        
        Returns:
            Dictionary with risk metrics or None if not applicable
        """
        if not self.config_manager.is_futures_market() or not self.futures_risk_manager:
            return None
            
        return self.futures_risk_manager.risk_metrics