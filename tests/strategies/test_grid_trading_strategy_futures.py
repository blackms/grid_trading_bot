import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from strategies.grid_trading_strategy import GridTradingStrategy
from config.trading_mode import TradingMode
from config.market_type import MarketType
from core.bot_management.event_bus import EventBus
from config.config_manager import ConfigManager
from core.services.exchange_interface import ExchangeInterface
from core.grid_management.grid_manager import GridManager
from core.order_handling.order_manager import OrderManager
from core.order_handling.balance_tracker import BalanceTracker
from core.order_handling.funding_rate_tracker import FundingRateTracker, FundingEvents
from core.order_handling.futures_position_manager import FuturesPositionManager, PositionEvents
from core.order_handling.futures_risk_manager import FuturesRiskManager, RiskEvents
from core.order_handling.stop_loss_manager import StopLossManager, StopLossEvents
from strategies.trading_performance_analyzer import TradingPerformanceAnalyzer
from strategies.plotter import Plotter

class TestGridTradingStrategyFutures:
    @pytest.fixture
    def config_manager(self):
        mock_config = Mock(spec=ConfigManager)
        mock_config.get_trading_mode.return_value = TradingMode.LIVE
        mock_config.get_market_type.return_value = MarketType.FUTURES
        mock_config.is_futures_market.return_value = True
        mock_config.get_contract_type.return_value = "perpetual"
        mock_config.get_leverage.return_value = 5
        mock_config.get_margin_type.return_value = "isolated"
        mock_config.is_hedge_mode_enabled.return_value = False
        mock_config.is_funding_rate_auto_hedge_enabled.return_value = True
        mock_config.get_base_currency.return_value = "BTC"
        mock_config.get_quote_currency.return_value = "USDT"
        return mock_config
    
    @pytest.fixture
    def event_bus(self):
        mock_event_bus = Mock(spec=EventBus)
        mock_event_bus.subscribe = Mock()
        mock_event_bus.publish = AsyncMock()
        return mock_event_bus
    
    @pytest.fixture
    def exchange_service(self):
        mock_exchange = Mock(spec=ExchangeInterface)
        mock_exchange.get_current_price = AsyncMock(return_value=60000)
        mock_exchange.get_positions = AsyncMock(return_value=[])
        return mock_exchange
    
    @pytest.fixture
    def grid_manager(self):
        mock_grid_manager = Mock(spec=GridManager)
        mock_grid_manager.get_trigger_price.return_value = 60000
        return mock_grid_manager
    
    @pytest.fixture
    def order_manager(self):
        mock_order_manager = Mock(spec=OrderManager)
        mock_order_manager.perform_initial_purchase = AsyncMock()
        mock_order_manager.initialize_grid_orders = AsyncMock()
        mock_order_manager.execute_take_profit_or_stop_loss_order = AsyncMock()
        return mock_order_manager
    
    @pytest.fixture
    def balance_tracker(self):
        mock_balance_tracker = Mock(spec=BalanceTracker)
        mock_balance_tracker.get_total_balance_value.return_value = 10000
        mock_balance_tracker.crypto_balance = 0.1
        return mock_balance_tracker
    
    @pytest.fixture
    def trading_performance_analyzer(self):
        mock_analyzer = Mock(spec=TradingPerformanceAnalyzer)
        mock_analyzer.generate_performance_summary.return_value = ({}, [])
        return mock_analyzer
    
    @pytest.fixture
    def plotter(self):
        return Mock(spec=Plotter)
    
    @pytest.fixture
    def funding_rate_tracker(self):
        mock_tracker = Mock(spec=FundingRateTracker)
        mock_tracker.current_funding_rate = 0.0001
        mock_tracker.next_funding_time = datetime.now() + timedelta(hours=8)
        mock_tracker.forecast_funding_rates = AsyncMock(return_value=[])
        return mock_tracker
    
    @pytest.fixture
    def futures_position_manager(self):
        mock_position_manager = Mock(spec=FuturesPositionManager)
        mock_position_manager.get_position = AsyncMock(return_value=None)
        mock_position_manager.get_all_positions = AsyncMock(return_value=[])
        mock_position_manager.close_position = AsyncMock()
        return mock_position_manager
    
    @pytest.fixture
    def futures_risk_manager(self):
        mock_risk_manager = Mock(spec=FuturesRiskManager)
        mock_risk_manager.risk_metrics = {
            "liquidation_risk_level": 0.0,
            "margin_health": 1.0,
            "current_drawdown": 0.0
        }
        mock_risk_manager.is_position_within_risk_limits = AsyncMock(return_value=(True, {}))
        return mock_risk_manager
    
    @pytest.fixture
    def stop_loss_manager(self):
        mock_stop_loss = Mock(spec=StopLossManager)
        mock_stop_loss.stop_loss_metrics = {
            "current_usdt_loss": 0.0,
            "current_portfolio_loss_percentage": 0.0
        }
        return mock_stop_loss
    
    @pytest.fixture
    def strategy(self, config_manager, event_bus, exchange_service, grid_manager, 
                order_manager, balance_tracker, trading_performance_analyzer, 
                plotter, funding_rate_tracker, futures_position_manager,
                futures_risk_manager, stop_loss_manager):
        return GridTradingStrategy(
            config_manager=config_manager,
            event_bus=event_bus,
            exchange_service=exchange_service,
            grid_manager=grid_manager,
            order_manager=order_manager,
            balance_tracker=balance_tracker,
            trading_performance_analyzer=trading_performance_analyzer,
            trading_mode=TradingMode.LIVE,
            trading_pair="BTC/USDT",
            plotter=plotter,
            funding_rate_tracker=funding_rate_tracker,
            futures_position_manager=futures_position_manager,
            futures_risk_manager=futures_risk_manager,
            stop_loss_manager=stop_loss_manager
        )
    
    def test_initialization(self, strategy):
        # Verify futures components are initialized
        assert strategy.is_futures_market is True
        assert strategy.funding_rate_tracker is not None
        assert strategy.futures_position_manager is not None
        assert strategy.futures_risk_manager is not None
        assert strategy.stop_loss_manager is not None
    
    def test_initialize_strategy(self, strategy):
        # Call initialize_strategy
        strategy.initialize_strategy()
        
        # Verify futures-specific parameters are initialized
        assert strategy.leverage == 5
        assert strategy.margin_type == "isolated"
        assert strategy.contract_type == "perpetual"
        assert strategy.hedge_mode is False
        assert hasattr(strategy, "active_positions")
        assert hasattr(strategy, "position_history")
        assert hasattr(strategy, "risk_metrics_history")
        assert hasattr(strategy, "funding_payments")
        assert hasattr(strategy, "funding_rate_history")
    
    @pytest.mark.asyncio
    async def test_handle_funding_rate_update(self, strategy):
        # Create test data
        funding_data = {
            "pair": "BTC/USDT",
            "funding_rate": 0.0001,
            "next_funding_time": datetime.now().isoformat(),
            "timestamp": datetime.now().isoformat()
        }
        
        # Call handler
        await strategy._handle_funding_rate_update(funding_data)
        
        # Verify funding rate history is updated
        assert len(strategy.funding_rate_history) == 1
        assert strategy.funding_rate_history[0] == funding_data
    
    @pytest.mark.asyncio
    async def test_handle_upcoming_funding(self, strategy):
        # Create test data
        funding_data = {
            "pair": "BTC/USDT",
            "funding_rate": 0.0001,
            "funding_time": datetime.now().isoformat(),
            "time_to_funding_minutes": 15,
            "estimated_payment": 0.5,
            "will_pay": False,
            "timestamp": datetime.now().isoformat()
        }
        
        # Call handler
        await strategy._handle_upcoming_funding(funding_data)
        
        # No assertions needed as this is primarily a logging function
        # We're just verifying it doesn't raise exceptions
    
    @pytest.mark.asyncio
    async def test_handle_funding_trend_change(self, strategy):
        # Create test data
        trend_data = {
            "pair": "BTC/USDT",
            "trend_direction": "increasing",
            "short_term_average": 0.0002,
            "long_term_average": 0.0001,
            "recent_rates": [0.0001, 0.00015, 0.0002],
            "timestamp": datetime.now().isoformat()
        }
        
        # Call handler
        await strategy._handle_funding_trend_change(trend_data)
        
        # No assertions needed as this is primarily a logging function
        # We're just verifying it doesn't raise exceptions
    
    @pytest.mark.asyncio
    async def test_handle_liquidation_warning(self, strategy):
        # Create test data
        liquidation_data = {
            "pair": "BTC/USDT",
            "position_side": "long",
            "current_price": 60000,
            "liquidation_price": 55000,
            "distance_to_liquidation": 0.08,
            "threshold": 0.1,
            "position": {"size": 0.1, "entry_price": 60000}
        }
        
        # Call handler
        await strategy._handle_liquidation_warning(liquidation_data)
        
        # No assertions needed as this is primarily a logging function
        # We're just verifying it doesn't raise exceptions
    
    @pytest.mark.asyncio
    async def test_check_funding_rate_strategy_adjustments(self, strategy, funding_rate_tracker, futures_position_manager):
        # Setup
        funding_rate_tracker.current_funding_rate = 0.002  # High positive funding rate
        futures_position_manager.get_position.return_value = {
            "pair": "BTC/USDT",
            "position_side": "long",
            "size": 0.1,
            "entry_price": 60000
        }
        
        # Call method
        await strategy._check_funding_rate_strategy_adjustments(60000)
        
        # Verify auto-hedge logic was considered
        futures_position_manager.get_position.assert_awaited_with("long")
    
    @pytest.mark.asyncio
    async def test_check_risk_based_strategy_adjustments(self, strategy, futures_risk_manager):
        # Setup
        futures_risk_manager.risk_metrics = {
            "liquidation_risk_level": 0.8,  # High liquidation risk
            "margin_health": 0.1  # Low margin health
        }
        
        # Call method
        await strategy._check_risk_based_strategy_adjustments(60000)
        
        # No assertions needed as this is primarily a logging function with conditional logic
        # We're just verifying it doesn't raise exceptions
    
    @pytest.mark.asyncio
    async def test_handle_circuit_breaker(self, strategy):
        # Create test data
        circuit_breaker_data = {
            "pair": "BTC/USDT",
            "current_price": 60000,
            "price_change": 0.06,
            "threshold": 0.05,
            "cooldown_period": 300,
            "triggers_24h": 1,
            "max_triggers": 3
        }
        
        # Call handler
        await strategy._handle_circuit_breaker(circuit_breaker_data)
        
        # No assertions needed as this is primarily a logging function
        # We're just verifying it doesn't raise exceptions
    
    @pytest.mark.asyncio
    async def test_handle_drawdown_exceeded(self, strategy):
        # Create test data
        drawdown_data = {
            "current_drawdown": 0.15,
            "threshold": 0.1,
            "auto_close_enabled": True,
            "critical_threshold": 0.2
        }
        
        # Call handler
        await strategy._handle_drawdown_exceeded(drawdown_data)
        
        # No assertions needed as this is primarily a logging function
        # We're just verifying it doesn't raise exceptions
    
    @pytest.mark.asyncio
    async def test_handle_margin_health_warning(self, strategy):
        # Create test data
        margin_data = {
            "available_margin": 500,
            "used_margin": 1000,
            "margin_ratio": 0.5,
            "auto_reduce_enabled": True,
            "auto_reduce_percentage": 25
        }
        
        # Call handler
        await strategy._handle_margin_health_warning(margin_data)
        
        # No assertions needed as this is primarily a logging function
        # We're just verifying it doesn't raise exceptions
    
    @pytest.mark.asyncio
    async def test_handle_stop_loss_triggered(self, strategy):
        # Create test data
        stop_loss_data = {
            "pair": "BTC/USDT",
            "position_side": "long",
            "reason": "USDT-based stop loss triggered",
            "position_id": "BTC/USDT-long"
        }
        
        # Setup active positions
        strategy.active_positions = {
            "BTC/USDT-long": {
                "side": "long",
                "size": 0.1,
                "entry_price": 60000,
                "open_time": datetime.now(),
                "stop_loss_triggered": False
            }
        }
        
        # Call handler
        await strategy._handle_stop_loss_triggered(stop_loss_data)
        
        # Verify position is marked as stop loss triggered
        assert strategy.active_positions["BTC/USDT-long"]["stop_loss_triggered"] is True
    
    @pytest.mark.asyncio
    async def test_handle_stop_loss_warning(self, strategy):
        # Create test data
        warning_data = {
            "stop_loss_type": "usdt_per_position",
            "current_loss": 500,
            "max_loss": 1000,
            "warning_threshold": 0.7
        }
        
        # Call handler
        await strategy._handle_stop_loss_warning(warning_data)
        
        # No assertions needed as this is primarily a logging function
        # We're just verifying it doesn't raise exceptions
    
    def test_get_funding_rate_summary(self, strategy, funding_rate_tracker):
        # Setup
        strategy.funding_rate_history = [{"rate": 0.0001}]
        strategy.funding_payments = [{"amount": 0.5}]
        
        # Get summary
        summary = strategy.get_funding_rate_summary()
        
        # Verify summary
        assert summary["enabled"] is True
        assert summary["current_rate"] == funding_rate_tracker.current_funding_rate
        assert "next_funding_time" in summary
        assert summary["funding_history_count"] == 1
        assert summary["funding_payments_count"] == 1
    
    def test_get_risk_metrics_summary(self, strategy, futures_risk_manager):
        # Get summary
        summary = strategy.get_risk_metrics_summary()
        
        # Verify summary
        assert summary["enabled"] is True
        assert summary["liquidation_risk_level"] == futures_risk_manager.risk_metrics["liquidation_risk_level"]
        assert summary["margin_health"] == futures_risk_manager.risk_metrics["margin_health"]
        assert "current_drawdown" in summary
        assert "max_drawdown" in summary
        assert "circuit_breaker_active" in summary
    
    def test_get_position_summary(self, strategy):
        # Setup
        strategy.active_positions = {"BTC/USDT-long": {}}
        strategy.position_history = [{}]
        
        # Get summary
        summary = strategy.get_position_summary()
        
        # Verify summary
        assert summary["enabled"] is True
        assert summary["active_positions_count"] == 1
        assert summary["position_history_count"] == 1
        assert summary["leverage"] == 5
        assert summary["margin_type"] == "isolated"
        assert summary["contract_type"] == "perpetual"
        assert summary["hedge_mode"] is False
    
    @pytest.mark.asyncio
    async def test_open_futures_position(self, strategy, futures_position_manager, futures_risk_manager):
        # Setup
        futures_risk_manager.is_position_within_risk_limits.return_value = (True, {})
        
        # Open position
        result = await strategy.open_futures_position("long", 0.1, 60000)
        
        # Verify result
        assert result["success"] is True
        assert "position_id" in result
        
        # Verify position is tracked
        assert len(strategy.active_positions) == 1
    
    @pytest.mark.asyncio
    async def test_open_futures_position_exceeds_risk_limits(self, strategy, futures_risk_manager):
        # Setup
        futures_risk_manager.is_position_within_risk_limits.return_value = (False, {"max_size": 0.05})
        
        # Open position
        result = await strategy.open_futures_position("long", 0.1, 60000)
        
        # Verify result
        assert result["success"] is False
        assert "error" in result
        assert "risk_info" in result
    
    @pytest.mark.asyncio
    async def test_close_futures_position(self, strategy, futures_position_manager):
        # Setup
        position_id = "BTC/USDT-long"
        strategy.active_positions = {
            position_id: {
                "side": "long",
                "size": 0.1,
                "entry_price": 60000,
                "open_time": datetime.now(),
                "stop_loss_triggered": False
            }
        }
        futures_position_manager.close_position.return_value = {"success": True}
        
        # Close position
        result = await strategy.close_futures_position(position_id)
        
        # Verify result
        assert result["success"] is True
        
        # Verify position is moved to history
        assert len(strategy.active_positions) == 0
        assert len(strategy.position_history) == 1
    
    @pytest.mark.asyncio
    async def test_close_futures_position_partial(self, strategy, futures_position_manager):
        # Setup
        position_id = "BTC/USDT-long"
        strategy.active_positions = {
            position_id: {
                "side": "long",
                "size": 0.1,
                "entry_price": 60000,
                "open_time": datetime.now(),
                "stop_loss_triggered": False
            }
        }
        futures_position_manager.close_position.return_value = {"success": True}
        
        # Close position partially (50%)
        result = await strategy.close_futures_position(position_id, 50.0)
        
        # Verify result
        assert result["success"] is True
        
        # Verify position size is reduced
        assert strategy.active_positions[position_id]["size"] == 0.05
        assert len(strategy.position_history) == 0