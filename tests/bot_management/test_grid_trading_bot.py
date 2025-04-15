import pytest, os
from unittest.mock import Mock, AsyncMock, patch, MagicMock, ANY
from datetime import datetime
from config.config_manager import ConfigManager
from core.bot_management.grid_trading_bot import GridTradingBot
from core.bot_management.event_bus import EventBus
from core.bot_management.notification.notification_handler import NotificationHandler
from core.services.exceptions import UnsupportedExchangeError, DataFetchError, UnsupportedTimeframeError
from config.trading_mode import TradingMode

@pytest.fixture(autouse=True)
def mock_env_vars():
    with patch.dict(os.environ, {"EXCHANGE_API_KEY": "test_api_key", "EXCHANGE_SECRET_KEY": "test_secret_key"}):
        yield

class TestGridTradingBot:
    @pytest.fixture
    def config_manager(self):
        mock_config = Mock(spec=ConfigManager)
        mock_config.get_trading_mode.return_value = TradingMode.LIVE
        mock_config.get_initial_balance.return_value = 1000
        mock_config.get_exchange_name.return_value = "binance"
        mock_config.get_spacing_type.return_value = "arithmetic"
        mock_config.get_top_range.return_value = 2000
        mock_config.get_bottom_range.return_value = 1500
        mock_config.get_num_grids.return_value = 10
        mock_config.get_market_type.return_value = None
        mock_config.is_futures_market.return_value = False
        return mock_config

    @pytest.fixture
    def mock_event_bus(self):
        event_bus = Mock(spec=EventBus)
        event_bus.subscribe = Mock()
        event_bus.publish_sync = Mock()
        return event_bus

    @pytest.fixture
    def notification_handler(self):
        return Mock(spec=NotificationHandler)

    @pytest.fixture
    def bot(self, config_manager, notification_handler, mock_event_bus):
        return GridTradingBot(
            config_path="config.json",
            config_manager=config_manager,
            notification_handler=notification_handler,
            event_bus=mock_event_bus,
            save_performance_results_path="results.json",
            no_plot=True
        )

    @patch("core.bot_management.grid_trading_bot.ExchangeServiceFactory.create_exchange_service", side_effect=UnsupportedExchangeError("Unsupported Exchange"))
    def test_initialization_with_unsupported_exchange_error(self, mock_exchange_service, config_manager, notification_handler, mock_event_bus):
        with patch("core.bot_management.grid_trading_bot.logging.getLogger") as mock_logger:
            logger_instance = Mock()
            mock_logger.return_value = logger_instance

            with pytest.raises(UnsupportedExchangeError, match="Unsupported Exchange"):
                GridTradingBot("config.json", config_manager, notification_handler, mock_event_bus)

            logger_instance.error.assert_called_once_with("UnsupportedExchangeError: Unsupported Exchange")

    @patch("core.bot_management.grid_trading_bot.ExchangeServiceFactory.create_exchange_service", side_effect=DataFetchError("Data Fetch Error"))
    def test_initialization_with_data_fetch_error(self, mock_exchange_service, config_manager, notification_handler, mock_event_bus):
        with patch("core.bot_management.grid_trading_bot.logging.getLogger") as mock_logger:
            logger_instance = Mock()
            mock_logger.return_value = logger_instance

            with pytest.raises(DataFetchError, match="Data Fetch Error"):
                GridTradingBot("config.json", config_manager, notification_handler, mock_event_bus)

            logger_instance.error.assert_called_once_with("DataFetchError: Data Fetch Error")

    @patch("core.bot_management.grid_trading_bot.ExchangeServiceFactory.create_exchange_service", side_effect=UnsupportedTimeframeError("Unsupported Timeframe"))
    def test_initialization_with_unsupported_timeframe_error(self, mock_exchange_service, config_manager, notification_handler, mock_event_bus):
        with patch("core.bot_management.grid_trading_bot.logging.getLogger") as mock_logger:
            logger_instance = Mock()
            mock_logger.return_value = logger_instance

            with pytest.raises(UnsupportedTimeframeError, match="Unsupported Timeframe"):
                GridTradingBot("config.json", config_manager, notification_handler, mock_event_bus)

            logger_instance.error.assert_called_once_with("UnsupportedTimeframeError: Unsupported Timeframe")

    @patch("core.bot_management.grid_trading_bot.ExchangeServiceFactory.create_exchange_service", side_effect=Exception("Unexpected Error"))
    def test_initialization_with_unexpected_exception(self, mock_exchange_service, config_manager, notification_handler, mock_event_bus):
        with patch("core.bot_management.grid_trading_bot.logging.getLogger") as mock_logger:
            logger_instance = Mock()
            mock_logger.return_value = logger_instance

            with pytest.raises(Exception, match="Unexpected Error"):
                GridTradingBot("config.json", config_manager, notification_handler, mock_event_bus)

            logger_instance.error.assert_any_call("An unexpected error occurred.")
            logger_instance.error.assert_any_call(ANY)

    def test_initialization_with_missing_config(self, notification_handler, mock_event_bus):
        config_manager = Mock(spec=ConfigManager)
        config_manager.get_trading_mode.side_effect = AttributeError("Missing configuration")

        with patch("core.bot_management.grid_trading_bot.logging.getLogger") as mock_logger:
            logger_instance = Mock()
            mock_logger.return_value = logger_instance

            with pytest.raises(AttributeError, match="Missing configuration"):
                GridTradingBot("config.json", config_manager, notification_handler, mock_event_bus)

            logger_instance.error.assert_any_call("An unexpected error occurred.")
            logger_instance.error.assert_any_call(ANY)

    @pytest.mark.asyncio
    async def test_get_bot_health_status(self, bot):
        bot._check_strategy_health = AsyncMock(return_value=True)
        bot._get_exchange_status = AsyncMock(return_value="ok")

        health_status = await bot.get_bot_health_status()

        assert health_status["strategy"] is True
        assert health_status["exchange_status"] == "ok"
        assert health_status["overall"] is True

    @pytest.mark.asyncio
    async def test_is_healthy_strategy_stopped(self, bot):
        bot.strategy = Mock()
        bot.strategy._running = False
        bot.exchange_service.get_exchange_status = AsyncMock(return_value={"status": "ok"})

        health_status = await bot.get_bot_health_status()

        assert health_status["strategy"] is False
        assert health_status["exchange_status"] == "ok"
        assert health_status["overall"] is False

    @patch("core.bot_management.grid_trading_bot.GridTradingBot._generate_and_log_performance")
    @pytest.mark.asyncio
    async def test_generate_and_log_performance_direct(self, mock_performance, bot):
        mock_performance.return_value = {
            "config": bot.config_path,
            "performance_summary": {"profit": 100},
            "orders": []
        }

        result = bot._generate_and_log_performance()

        assert result == {
            "config": bot.config_path,
            "performance_summary": {"profit": 100},
            "orders": []
        }

    @pytest.mark.asyncio
    async def test_get_exchange_status(self, bot):
        bot.exchange_service = MagicMock()
        bot.exchange_service.get_exchange_status = AsyncMock(return_value={"status": "ok"})

        result = await bot._get_exchange_status()
        assert result == "ok"

    def test_get_balance_zero_values(self, bot):
        bot.balance_tracker = Mock()
        bot.balance_tracker.balance = 0.0
        bot.balance_tracker.reserved_fiat = 0.0
        bot.balance_tracker.crypto_balance = 0.0
        bot.balance_tracker.reserved_crypto = 0.0

        result = bot.get_balances()

        assert result == {
            "fiat": 0.0,
            "crypto": 0.0,
            "reserved_fiat": 0.0,
            "reserved_crypto": 0.0,
        }
    
    @patch("core.bot_management.grid_trading_bot.GridTradingStrategy")
    @pytest.mark.asyncio
    async def test_run_strategy_with_exception(self, mock_strategy, bot):
        # Arrange: Mock dependencies
        bot.balance_tracker = Mock()
        bot.balance_tracker.setup_balances = AsyncMock()
        bot.order_status_tracker.start_tracking = Mock()
        bot.strategy.run = AsyncMock(side_effect=Exception("Test Exception"))
        mock_grid_manager = Mock()
        mock_grid_manager.initialize_grids_and_levels = Mock()
        bot.strategy.grid_manager = mock_grid_manager

        with patch.object(bot.logger, "error") as mock_logger_error:
            # Act and Assert: Ensure the bot logs the error and re-raises the exception
            with pytest.raises(Exception, match="Test Exception"):
                await bot.run()

            # Ensure the error is logged correctly
            mock_logger_error.assert_any_call("An unexpected error occurred Test Exception")
            assert bot.is_running is False

    @patch("core.bot_management.grid_trading_bot.OrderStatusTracker")
    @pytest.mark.asyncio
    async def test_stop_bot(self, mock_order_status_tracker, bot):
        bot.is_running = True
        bot.strategy.stop = AsyncMock()
        bot.order_status_tracker.stop_tracking = AsyncMock()

        await bot._stop()

        bot.strategy.stop.assert_awaited_once()
        bot.order_status_tracker.stop_tracking.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_restart_bot(self, bot):
        bot.is_running = True
        bot.strategy.restart = AsyncMock()
        bot.order_status_tracker.start_tracking = Mock()
        bot._stop = AsyncMock()

        await bot.restart()

        bot._stop.assert_awaited_once()
        bot.strategy.restart.assert_awaited_once()
        bot.order_status_tracker.start_tracking.assert_called_once()

    @pytest.mark.asyncio
    async def test_restart_when_not_running(self, bot):
        bot.is_running = False
        bot._stop = AsyncMock()
        bot.strategy.restart = AsyncMock()
        bot.order_status_tracker.start_tracking = Mock()

        await bot.restart()

        bot._stop.assert_not_awaited()
        bot.strategy.restart.assert_awaited_once()
        bot.order_status_tracker.start_tracking.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_stop_bot_event(self, bot):
        bot.is_running = True
        bot._stop = AsyncMock()

        await bot._handle_stop_bot_event("Test reason")

        bot._stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_handle_stop_bot_event_when_running(self, bot):
        bot.is_running = True

        # Mock _stop to simulate state change
        async def mock_stop():
            bot.is_running = False

        bot._stop = AsyncMock(side_effect=mock_stop)

        await bot._handle_stop_bot_event("Test reason")

        bot._stop.assert_awaited_once_with()
        assert not bot.is_running

    @pytest.mark.asyncio
    async def test_handle_stop_bot_event_when_already_stopped(self, bot):
        # Arrange: Mock dependencies and state
        bot.is_running = False
        bot._stop = AsyncMock()

        # Act: Call the method
        await bot._handle_stop_bot_event("Test reason")

        # Assert: Verify that _stop was called but exited early
        bot._stop.assert_awaited_once()
        bot._stop.assert_awaited_once_with()  # Ensures _stop was called without additional arguments

    @pytest.mark.asyncio
    async def test_handle_start_bot_event(self, bot):
        bot.is_running = False
        bot.restart = AsyncMock()

        await bot._handle_start_bot_event("Test reason")

        bot.restart.assert_awaited_once()
    
    @patch("core.bot_management.grid_trading_bot.GridTradingStrategy")
    @pytest.mark.asyncio
    async def test_run_with_plotting_enabled(self, mock_strategy, bot):
        bot.no_plot = False
        bot.balance_tracker.setup_balances = AsyncMock()
        bot._generate_and_log_performance = Mock()
        bot.strategy.plot_results = Mock()
        bot.strategy.run = AsyncMock()
        bot.order_status_tracker.start_tracking = Mock()
        mock_grid_manager = Mock()
        mock_grid_manager.initialize_grids_and_levels = Mock()
        bot.strategy.grid_manager = mock_grid_manager

        await bot.run()

        bot.strategy.plot_results.assert_called_once()
        bot.strategy.run.assert_awaited_once()
        bot.order_status_tracker.start_tracking.assert_called_once()
        
    @pytest.fixture
    def futures_config_manager(self):
        mock_config = Mock(spec=ConfigManager)
        mock_config.get_trading_mode.return_value = TradingMode.LIVE
        mock_config.get_initial_balance.return_value = 1000
        mock_config.get_exchange_name.return_value = "bybit"
        mock_config.get_spacing_type.return_value = "arithmetic"
        mock_config.get_top_range.return_value = 2000
        mock_config.get_bottom_range.return_value = 1500
        mock_config.get_num_grids.return_value = 10
        
        # Futures-specific configuration
        from config.market_type import MarketType
        mock_config.get_market_type.return_value = MarketType.FUTURES
        mock_config.is_futures_market.return_value = True
        mock_config.get_contract_type.return_value = "perpetual"
        mock_config.get_leverage.return_value = 5
        mock_config.get_margin_type.return_value = "isolated"
        mock_config.is_hedge_mode_enabled.return_value = False
        mock_config.is_liquidation_protection_enabled.return_value = True
        mock_config.get_liquidation_protection_threshold.return_value = 0.3
        mock_config.is_dynamic_grid_enabled.return_value = True
        mock_config.is_funding_rate_monitoring_enabled.return_value = True
        mock_config.is_stop_loss_manager_enabled.return_value = True
        
        return mock_config
    
    @pytest.fixture
    def futures_bot(self, futures_config_manager, notification_handler, mock_event_bus):
        with patch("core.bot_management.grid_trading_bot.FuturesPositionManager") as mock_position_manager, \
             patch("core.bot_management.grid_trading_bot.FuturesRiskManager") as mock_risk_manager, \
             patch("core.bot_management.grid_trading_bot.FundingRateTracker") as mock_funding_tracker, \
             patch("core.bot_management.grid_trading_bot.DynamicGridManager") as mock_dynamic_grid, \
             patch("core.bot_management.grid_trading_bot.StopLossManager") as mock_stop_loss:
            
            return GridTradingBot(
                config_path="futures_config.json",
                config_manager=futures_config_manager,
                notification_handler=notification_handler,
                event_bus=mock_event_bus,
                save_performance_results_path="futures_results.json",
                no_plot=True
            )
    
    @pytest.mark.asyncio
    async def test_futures_bot_initialization(self, futures_bot):
        # Verify futures components are initialized
        assert futures_bot.futures_position_manager is not None
        assert futures_bot.futures_risk_manager is not None
        assert futures_bot.funding_rate_tracker is not None
        assert futures_bot.dynamic_grid_manager is not None
        assert futures_bot.stop_loss_manager is not None
    
    @pytest.mark.asyncio
    async def test_futures_bot_run(self, futures_bot):
        # Mock dependencies
        futures_bot.balance_tracker = Mock()
        futures_bot.balance_tracker.setup_balances = AsyncMock()
        futures_bot.order_status_tracker.start_tracking = Mock()
        futures_bot.strategy.run = AsyncMock()
        futures_bot.strategy.initialize_strategy = Mock()
        futures_bot._generate_and_log_performance = Mock()
        
        # Mock futures components
        futures_bot.futures_position_manager.initialize = AsyncMock()
        futures_bot.futures_risk_manager.initialize = AsyncMock()
        futures_bot.funding_rate_tracker.initialize = AsyncMock()
        futures_bot.dynamic_grid_manager.initialize_dynamic_grids = AsyncMock()
        futures_bot.stop_loss_manager.initialize = AsyncMock()
        
        # Run the bot
        await futures_bot.run()
        
        # Verify futures components are initialized
        futures_bot.futures_position_manager.initialize.assert_awaited_once()
        futures_bot.futures_risk_manager.initialize.assert_awaited_once()
        futures_bot.funding_rate_tracker.initialize.assert_awaited_once()
        futures_bot.dynamic_grid_manager.initialize_dynamic_grids.assert_awaited_once()
        futures_bot.stop_loss_manager.initialize.assert_awaited_once()
        
        # Verify strategy is run
        futures_bot.strategy.run.assert_awaited_once()
    
    @pytest.mark.asyncio
    async def test_futures_bot_stop(self, futures_bot):
        # Mock dependencies
        futures_bot.is_running = True
        futures_bot.strategy.stop = AsyncMock()
        futures_bot.order_status_tracker.stop_tracking = AsyncMock()
        
        # Mock futures components
        futures_bot.futures_position_manager.shutdown = AsyncMock()
        futures_bot.futures_risk_manager.shutdown = AsyncMock()
        futures_bot.funding_rate_tracker.shutdown = AsyncMock()
        futures_bot.dynamic_grid_manager.shutdown = AsyncMock()
        futures_bot.stop_loss_manager.shutdown = AsyncMock()
        
        # Stop the bot
        await futures_bot._stop()
        
        # Verify futures components are shut down
        futures_bot.funding_rate_tracker.shutdown.assert_awaited_once()
        futures_bot.stop_loss_manager.shutdown.assert_awaited_once()
        futures_bot.dynamic_grid_manager.shutdown.assert_awaited_once()
        futures_bot.futures_risk_manager.shutdown.assert_awaited_once()
        futures_bot.futures_position_manager.shutdown.assert_awaited_once()
        
        # Verify strategy is stopped
        futures_bot.strategy.stop.assert_awaited_once()
        futures_bot.order_status_tracker.stop_tracking.assert_awaited_once()
        
        # Verify bot is no longer running
        assert not futures_bot.is_running
    
    @pytest.mark.asyncio
    async def test_futures_bot_health_status(self, futures_bot):
        # Mock dependencies
        futures_bot._check_strategy_health = AsyncMock(return_value=True)
        futures_bot._get_exchange_status = AsyncMock(return_value="ok")
        
        # Get health status
        health_status = await futures_bot.get_bot_health_status()
        
        # Verify health status includes futures components
        assert health_status["strategy"] is True
        assert health_status["exchange_status"] == "ok"
        assert "futures_position_manager" in health_status
        assert "futures_risk_manager" in health_status
        assert "funding_rate_tracker" in health_status
        assert "dynamic_grid_manager" in health_status
        assert "stop_loss_manager" in health_status
        assert health_status["overall"] is True
    
    @pytest.mark.asyncio
    async def test_futures_bot_get_balances(self, futures_bot):
        # Mock dependencies
        futures_bot.balance_tracker = Mock()
        futures_bot.balance_tracker.balance = 1000.0
        futures_bot.balance_tracker.reserved_fiat = 200.0
        futures_bot.balance_tracker.crypto_balance = 0.5
        futures_bot.balance_tracker.reserved_crypto = 0.1
        
        # Mock futures components
        futures_bot.futures_position_manager.get_all_positions = Mock(return_value=[
            {"pair": "BTC/USDT", "position_side": "long", "size": 0.1, "entry_price": 60000}
        ])
        futures_bot.funding_rate_tracker.current_funding_rate = 0.0001
        futures_bot.funding_rate_tracker.next_funding_time = datetime.now()
        futures_bot.futures_risk_manager.risk_metrics = {"liquidation_risk_level": 0.1}
        futures_bot.stop_loss_manager.stop_loss_metrics = {"current_usdt_loss": 0.0}
        
        # Get balances
        balances = futures_bot.get_balances()
        
        # Verify balances include futures information
        assert balances["fiat"] == 1000.0
        assert balances["reserved_fiat"] == 200.0
        assert balances["crypto"] == 0.5
        assert balances["reserved_crypto"] == 0.1
        assert "futures" in balances
        assert "positions" in balances["futures"]
        assert "current_funding_rate" in balances["futures"]
        assert "next_funding_time" in balances["futures"]
        assert "risk_metrics" in balances["futures"]
        assert "stop_loss_metrics" in balances["futures"]
    
    @pytest.mark.asyncio
    async def test_get_funding_rate_info(self, futures_bot):
        # Mock funding rate tracker
        futures_bot.funding_rate_tracker.get_current_funding_info = AsyncMock(return_value={
            "current_rate": 0.0001,
            "next_funding_time": "2025-04-15T08:00:00"
        })
        
        # Get funding rate info
        funding_info = await futures_bot.get_funding_rate_info()
        
        # Verify funding rate info
        assert funding_info is not None
        assert "current_rate" in funding_info
        assert "next_funding_time" in funding_info
    
    @pytest.mark.asyncio
    async def test_get_funding_rate_forecast(self, futures_bot):
        # Mock funding rate tracker
        futures_bot.funding_rate_tracker.forecast_funding_rates = AsyncMock(return_value=[
            {"timestamp": "2025-04-15T08:00:00", "forecasted_rate": 0.0001, "confidence": 0.8}
        ])
        
        # Get funding rate forecast
        forecast = await futures_bot.get_funding_rate_forecast()
        
        # Verify forecast
        assert forecast is not None
        assert len(forecast) == 1
        assert "timestamp" in forecast[0]
        assert "forecasted_rate" in forecast[0]
        assert "confidence" in forecast[0]
    
    @pytest.mark.asyncio
    async def test_get_stop_loss_metrics(self, futures_bot):
        # Mock stop loss manager
        futures_bot.stop_loss_manager.get_stop_loss_metrics = AsyncMock(return_value={
            "current_usdt_loss": 0.0,
            "current_portfolio_loss_percentage": 0.0
        })
        
        # Get stop loss metrics
        metrics = await futures_bot.get_stop_loss_metrics()
        
        # Verify metrics
        assert metrics is not None
        assert "current_usdt_loss" in metrics
        assert "current_portfolio_loss_percentage" in metrics
    
    @pytest.mark.asyncio
    async def test_update_stop_loss_settings(self, futures_bot):
        # Mock stop loss manager
        futures_bot.stop_loss_manager.update_stop_loss_settings = AsyncMock()
        
        # Update stop loss settings
        new_settings = {"usdt_stop_loss": {"max_loss_amount": 500.0}}
        result = await futures_bot.update_stop_loss_settings(new_settings)
        
        # Verify settings were updated
        futures_bot.stop_loss_manager.update_stop_loss_settings.assert_awaited_once_with(new_settings)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_get_risk_metrics(self, futures_bot):
        # Mock risk manager
        futures_bot.futures_risk_manager.risk_metrics = {
            "liquidation_risk_level": 0.1,
            "margin_health": 0.9
        }
        
        # Get risk metrics
        metrics = await futures_bot.get_risk_metrics()
        
        # Verify metrics
        assert metrics is not None
        assert "liquidation_risk_level" in metrics
        assert "margin_health" in metrics