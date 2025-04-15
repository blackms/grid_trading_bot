import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from decimal import Decimal
import time

from config.config_manager import ConfigManager
from core.services.exchange_interface import ExchangeInterface
from core.bot_management.event_bus import EventBus, Events
from core.order_handling.futures_position_manager import FuturesPositionManager, Position, PositionEvents
from core.order_handling.futures_risk_manager import FuturesRiskManager, RiskEvents

@pytest.fixture
def mock_config_manager():
    config_manager = MagicMock(spec=ConfigManager)
    config_manager.is_futures_market.return_value = True
    config_manager.get_base_currency.return_value = "BTC"
    config_manager.get_quote_currency.return_value = "USDT"
    config_manager.get_leverage.return_value = 5
    config_manager.get_margin_type.return_value = "isolated"
    config_manager.is_hedge_mode_enabled.return_value = False
    config_manager.is_liquidation_protection_enabled.return_value = True
    config_manager.get_liquidation_protection_threshold.return_value = 0.1
    config_manager.get_max_position_size.return_value = 10.0
    
    # Mock futures risk management settings
    futures_risk = {
        "circuit_breaker": {
            "enabled": True,
            "price_change_threshold": 0.1,
            "cooldown_period": 300,
            "max_daily_triggers": 3
        },
        "margin_health": {
            "warning_threshold": 0.5,
            "critical_threshold": 0.2,
            "auto_reduce_enabled": False
        },
        "funding_rate": {
            "high_threshold": 0.001,
            "extreme_threshold": 0.003,
            "cumulative_threshold": 0.01
        },
        "drawdown": {
            "warning_threshold": 0.1,
            "critical_threshold": 0.2,
            "max_drawdown": 0.3,
            "auto_close_enabled": False
        },
        "position_sizing": {
            "max_capital_allocation": 0.2,
            "scale_with_volatility": True,
            "dynamic_leverage": True
        }
    }
    
    config_manager.get_futures_risk_management.return_value = futures_risk
    config_manager.get_liquidation_protection.return_value = {"enabled": True, "threshold": 0.1}
    
    return config_manager

@pytest.fixture
def mock_exchange_service():
    exchange_service = AsyncMock(spec=ExchangeInterface)
    exchange_service.get_current_price.return_value = 50000.0
    exchange_service.get_funding_rate.return_value = {"fundingRate": 0.0001, "nextFundingTime": int(time.time()) + 28800}
    exchange_service.get_contract_specifications.return_value = {
        "contractSize": 1,
        "maintenanceMarginRate": 0.005,
        "tickSize": 0.5,
        "maxLeverage": 100
    }
    exchange_service.get_balance.return_value = {
        "USDT": {
            "free": 100000.0,
            "used": 10000.0,
            "total": 110000.0
        }
    }
    return exchange_service

@pytest.fixture
def mock_event_bus():
    event_bus = MagicMock(spec=EventBus)
    event_bus.publish = AsyncMock()
    event_bus.subscribe = AsyncMock()
    return event_bus

@pytest.fixture
def mock_position_manager():
    position_manager = AsyncMock(spec=FuturesPositionManager)
    position_manager.get_all_positions.return_value = []
    position_manager.close_position = AsyncMock()
    return position_manager

@pytest.fixture
async def risk_manager(mock_config_manager, mock_exchange_service, mock_event_bus, mock_position_manager):
    manager = FuturesRiskManager(
        config_manager=mock_config_manager,
        exchange_service=mock_exchange_service,
        event_bus=mock_event_bus,
        position_manager=mock_position_manager
    )
    return manager

@pytest.mark.asyncio
async def test_initialization(risk_manager):
    # Test initialization
    await risk_manager.initialize()
    
    # Verify event subscriptions
    assert risk_manager.event_bus.subscribe.call_count == 5
    
    # Verify risk metrics were initialized
    assert "last_price" in risk_manager.risk_metrics
    assert "current_funding_rate" in risk_manager.risk_metrics
    assert "liquidation_risk_level" in risk_manager.risk_metrics
    
    # Verify risk limits were loaded
    assert "circuit_breaker" in risk_manager.risk_limits
    assert "margin_health" in risk_manager.risk_limits
    assert "funding_rate" in risk_manager.risk_limits
    assert "drawdown" in risk_manager.risk_limits
    assert "position_sizing" in risk_manager.risk_limits

@pytest.mark.asyncio
async def test_liquidation_risk_detection(risk_manager, mock_position_manager, mock_exchange_service, mock_event_bus):
    # Initialize the risk manager
    await risk_manager.initialize()
    
    # Mock a position with liquidation risk
    position_data = {
        "pair": "BTCUSDT",
        "position_side": "long",
        "size": 1.0,
        "entry_price": 50000.0,
        "liquidation_price": 45000.0,  # Close to current price
        "unrealized_pnl": -2000.0
    }
    mock_position_manager.get_all_positions.return_value = [position_data]
    
    # Set current price close to liquidation price
    mock_exchange_service.get_current_price.return_value = 46000.0  # Only 2% away from liquidation
    
    # Manually trigger liquidation check
    await risk_manager._monitor_liquidation_risk()
    
    # Verify liquidation risk event was published
    mock_event_bus.publish.assert_called_with(
        RiskEvents.LIQUIDATION_RISK_DETECTED,
        {
            "pair": "BTCUSDT",
            "position_side": "long",
            "current_price": 46000.0,
            "liquidation_price": 45000.0,
            "distance_to_liquidation": (46000.0 - 45000.0) / 46000.0,  # About 2.17%
            "threshold": 0.1,
            "risk_factor": 1.0 - ((46000.0 - 45000.0) / 46000.0) / 0.1,  # About 0.78
            "recommended_reduction": 1.0 - ((46000.0 - 45000.0) / 46000.0) / 0.1,  # About 0.78
            "position": position_data
        }
    )

@pytest.mark.asyncio
async def test_circuit_breaker_trigger(risk_manager, mock_exchange_service, mock_event_bus):
    # Initialize the risk manager
    await risk_manager.initialize()
    
    # Set initial price
    mock_exchange_service.get_current_price.return_value = 50000.0
    
    # Manually trigger circuit breaker check to initialize price history
    await risk_manager._monitor_circuit_breakers()
    
    # Set price with large change
    mock_exchange_service.get_current_price.return_value = 40000.0  # 20% drop
    
    # Manually trigger circuit breaker check
    await risk_manager._monitor_circuit_breakers()
    
    # Verify circuit breaker event was published
    mock_event_bus.publish.assert_called_with(
        RiskEvents.CIRCUIT_BREAKER_TRIGGERED,
        {
            "pair": "BTC/USDT",
            "current_price": 40000.0,
            "price_change": 0.2,
            "threshold": 0.1,
            "cooldown_period": 300,
            "triggers_24h": 1,
            "max_triggers": 3
        }
    )
    
    # Verify circuit breaker was activated
    assert risk_manager.circuit_breaker_active is True

@pytest.mark.asyncio
async def test_margin_health_warning(risk_manager, mock_position_manager, mock_exchange_service, mock_event_bus):
    # Initialize the risk manager
    await risk_manager.initialize()
    
    # Mock a position
    position_data = {
        "pair": "BTCUSDT",
        "position_side": "long",
        "size": 1.0,
        "entry_price": 50000.0
    }
    mock_position_manager.get_all_positions.return_value = [position_data]
    
    # Set low available margin
    mock_exchange_service.get_balance.return_value = {
        "USDT": {
            "free": 2000.0,
            "used": 10000.0,
            "total": 12000.0
        }
    }
    
    # Manually trigger margin health check
    await risk_manager._monitor_margin_health()
    
    # Verify margin health warning event was published
    mock_event_bus.publish.assert_called_with(
        RiskEvents.MARGIN_HEALTH_WARNING,
        {
            "available_margin": 2000.0,
            "used_margin": 10000.0,
            "total_margin": 12000.0,
            "margin_ratio": 0.2,
            "warning_threshold": 0.5,
            "critical_threshold": 0.2
        }
    )

@pytest.mark.asyncio
async def test_funding_rate_alert(risk_manager, mock_position_manager, mock_exchange_service, mock_event_bus):
    # Initialize the risk manager
    await risk_manager.initialize()
    
    # Mock a position
    position_data = {
        "pair": "BTCUSDT",
        "position_side": "long",
        "size": 1.0,
        "entry_price": 50000.0
    }
    mock_position_manager.get_all_positions.return_value = [position_data]
    
    # Set high funding rate
    mock_exchange_service.get_funding_rate.return_value = {
        "fundingRate": 0.004,  # 0.4% - above extreme threshold
        "nextFundingTime": int(time.time()) + 28800
    }
    
    # Manually trigger funding rate check
    await risk_manager._monitor_funding_rates()
    
    # Verify funding rate alert event was published
    mock_event_bus.publish.assert_called_with(
        RiskEvents.FUNDING_RATE_ALERT,
        {
            "pair": "BTC/USDT",
            "current_funding_rate": 0.004,
            "threshold": 0.003,
            "next_funding_time": mock_exchange_service.get_funding_rate.return_value["nextFundingTime"],
            "estimated_impact": -200.0,  # 50000 * 0.004 = 200, negative for long position
            "alert_level": "extreme"
        }
    )

@pytest.mark.asyncio
async def test_drawdown_warning(risk_manager, mock_position_manager, mock_exchange_service, mock_event_bus):
    # Initialize the risk manager
    await risk_manager.initialize()
    
    # Set initial equity peak
    risk_manager.drawdown_tracking["peak_equity"] = 100000.0
    
    # Mock a position with unrealized loss
    position_data = {
        "pair": "BTCUSDT",
        "position_side": "long",
        "size": 1.0,
        "entry_price": 50000.0,
        "unrealized_pnl": -15000.0
    }
    mock_position_manager.get_all_positions.return_value = [position_data]
    
    # Set current equity
    mock_exchange_service.get_balance.return_value = {
        "USDT": {
            "free": 75000.0,
            "used": 10000.0,
            "total": 85000.0
        }
    }
    
    # Manually trigger drawdown check
    await risk_manager._monitor_drawdown()
    
    # Verify drawdown warning event was published
    mock_event_bus.publish.assert_called_with(
        RiskEvents.DRAWDOWN_THRESHOLD_EXCEEDED,
        {
            "current_equity": 70000.0,  # 85000 - 15000 = 70000
            "peak_equity": 100000.0,
            "current_drawdown": 0.3,  # (100000 - 70000) / 100000 = 0.3
            "warning_threshold": 0.1,
            "critical_threshold": 0.2,
            "drawdown_duration": pytest.approx(time.time() - risk_manager.drawdown_tracking["drawdown_start_time"], abs=1),
            "alert_level": "critical"
        }
    )

@pytest.mark.asyncio
async def test_position_sizing(risk_manager, mock_exchange_service):
    # Initialize the risk manager
    await risk_manager.initialize()
    
    # Calculate max position size
    max_size = await risk_manager.calculate_max_position_size("long", 50000.0)
    
    # Verify max position size calculation
    # Available balance: 100000.0
    # Max capital allocation: 20% = 20000.0
    # Leverage: 5x
    # Max position value: 20000.0 * 5 = 100000.0
    # Max position size: 100000.0 / 50000.0 = 2.0 BTC
    assert max_size == pytest.approx(2.0, abs=0.1)
    
    # Check if position is within risk limits
    is_within_limits, risk_info = await risk_manager.is_position_within_risk_limits("long", 1.5, 50000.0)
    
    # Verify position is within limits
    assert is_within_limits is True
    assert risk_info["utilization"] == pytest.approx(0.75, abs=0.1)
    
    # Check if position exceeds risk limits
    is_within_limits, risk_info = await risk_manager.is_position_within_risk_limits("long", 3.0, 50000.0)
    
    # Verify position exceeds limits
    assert is_within_limits is False
    assert risk_info["utilization"] > 1.0

@pytest.mark.asyncio
async def test_shutdown(risk_manager):
    # Initialize the risk manager
    await risk_manager.initialize()
    
    # Verify tasks were created
    assert len(risk_manager.risk_update_tasks) > 0
    
    # Shutdown the risk manager
    await risk_manager.shutdown()
    
    # Verify tasks were cancelled
    assert len(risk_manager.risk_update_tasks) == 0