import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from decimal import Decimal

from config.config_manager import ConfigManager
from core.services.exchange_interface import ExchangeInterface
from core.bot_management.event_bus import EventBus, Events
from core.order_handling.futures_position_manager import FuturesPositionManager, Position, PositionEvents
from core.order_handling.futures_risk_manager import FuturesRiskManager, RiskEvents
from core.order_handling.stop_loss_manager import StopLossManager, StopLossEvents

@pytest.fixture
def mock_config_manager():
    config_manager = MagicMock(spec=ConfigManager)
    config_manager.is_futures_market.return_value = True
    config_manager.get_base_currency.return_value = "BTC"
    config_manager.get_quote_currency.return_value = "USDT"
    config_manager.is_stop_loss_enabled.return_value = True
    config_manager.get_stop_loss_threshold.return_value = 0.05
    
    # Mock futures risk management settings
    futures_risk = {
        "usdt_stop_loss": {
            "enabled": True,
            "max_loss_amount": 1000.0,
            "per_position": False,
            "warning_threshold": 0.7
        },
        "portfolio_stop_loss": {
            "enabled": True,
            "max_loss_percentage": 0.1,
            "warning_threshold": 0.7
        }
    }
    config_manager.get_futures_risk_management.return_value = futures_risk
    
    return config_manager

@pytest.fixture
def mock_exchange_service():
    exchange_service = AsyncMock(spec=ExchangeInterface)
    
    # Mock balance response
    balance_data = {
        "USDT": {
            "free": 5000.0,
            "used": 5000.0,
            "total": 10000.0
        }
    }
    exchange_service.get_balance.return_value = balance_data
    
    # Mock current price
    exchange_service.get_current_price.return_value = 20000.0
    
    return exchange_service

@pytest.fixture
def mock_event_bus():
    event_bus = AsyncMock(spec=EventBus)
    return event_bus

@pytest.fixture
def mock_position_manager():
    position_manager = AsyncMock(spec=FuturesPositionManager)
    
    # Mock positions
    positions = [
        {
            "pair": "BTC/USDT",
            "position_side": "long",
            "size": 0.5,
            "entry_price": 19000.0,
            "leverage": 3,
            "margin_type": "isolated",
            "initial_margin": 3166.67,
            "liquidation_price": 17000.0,
            "unrealized_pnl": 500.0,
            "realized_pnl": 0.0,
            "position_id": "BTCUSDT-long",
            "is_open": True
        },
        {
            "pair": "BTC/USDT",
            "position_side": "short",
            "size": 0.3,
            "entry_price": 21000.0,
            "leverage": 3,
            "margin_type": "isolated",
            "initial_margin": 2100.0,
            "liquidation_price": 23000.0,
            "unrealized_pnl": 300.0,
            "realized_pnl": 0.0,
            "position_id": "BTCUSDT-short",
            "is_open": True
        }
    ]
    position_manager.get_all_positions.return_value = positions
    
    return position_manager

@pytest.fixture
def mock_risk_manager():
    risk_manager = AsyncMock(spec=FuturesRiskManager)
    return risk_manager

@pytest.fixture
def stop_loss_manager(mock_config_manager, mock_exchange_service, mock_event_bus, mock_position_manager, mock_risk_manager):
    manager = StopLossManager(
        config_manager=mock_config_manager,
        exchange_service=mock_exchange_service,
        event_bus=mock_event_bus,
        position_manager=mock_position_manager,
        risk_manager=mock_risk_manager
    )
    return manager

@pytest.mark.asyncio
async def test_initialization(stop_loss_manager, mock_event_bus):
    """Test that the StopLossManager initializes correctly."""
    await stop_loss_manager.initialize()
    
    # Verify that event subscriptions were set up
    assert mock_event_bus.subscribe.call_count >= 4
    
    # Verify that stop loss metrics were initialized
    assert "initial_portfolio_value" in stop_loss_manager.stop_loss_metrics
    assert "current_portfolio_value" in stop_loss_manager.stop_loss_metrics
    assert "max_portfolio_value" in stop_loss_manager.stop_loss_metrics
    
    # Verify that position tracking was set up
    assert "positions_with_stop_loss" in stop_loss_manager.stop_loss_metrics
    assert len(stop_loss_manager.stop_loss_metrics["positions_with_stop_loss"]) == 2

@pytest.mark.asyncio
async def test_usdt_stop_loss_warning(stop_loss_manager, mock_event_bus, mock_position_manager):
    """Test USDT-based stop loss warning."""
    # Initialize the manager
    await stop_loss_manager.initialize()
    
    # Mock a position with loss
    position = {
        "pair": "BTC/USDT",
        "position_side": "long",
        "size": 1.0,
        "entry_price": 20000.0,
        "unrealized_pnl": -700.0,  # 700 USDT loss
        "position_id": "BTCUSDT-long"
    }
    
    # Set up position manager to return this position
    mock_position_manager.get_all_positions.return_value = [position]
    
    # Call the USDT stop loss monitoring method directly
    await stop_loss_manager._monitor_usdt_stop_loss()
    
    # Verify that a warning event was published
    mock_event_bus.publish.assert_called_with(
        StopLossEvents.STOP_LOSS_WARNING,
        {
            "stop_loss_type": "usdt_total",
            "current_loss": 700.0,
            "max_loss": 1000.0,
            "warning_threshold": 700.0,
            "positions": [position]
        }
    )

@pytest.mark.asyncio
async def test_usdt_stop_loss_execution(stop_loss_manager, mock_event_bus, mock_position_manager):
    """Test USDT-based stop loss execution."""
    # Initialize the manager
    await stop_loss_manager.initialize()
    
    # Mock a position with loss exceeding threshold
    position = {
        "pair": "BTC/USDT",
        "position_side": "long",
        "size": 1.0,
        "entry_price": 20000.0,
        "unrealized_pnl": -1200.0,  # 1200 USDT loss, exceeds 1000 threshold
        "position_id": "BTCUSDT-long"
    }
    
    # Set up position manager to return this position
    mock_position_manager.get_all_positions.return_value = [position]
    
    # Call the USDT stop loss monitoring method directly
    await stop_loss_manager._monitor_usdt_stop_loss()
    
    # Verify that stop loss was executed
    mock_position_manager.close_position.assert_called_with(
        side="long",
        order_type="market"
    )
    
    # Verify that events were published
    assert mock_event_bus.publish.call_count >= 2
    
    # Check for STOP_LOSS_TRIGGERED event
    trigger_calls = [call for call in mock_event_bus.publish.call_args_list 
                    if call[0][0] == StopLossEvents.STOP_LOSS_TRIGGERED]
    assert len(trigger_calls) > 0
    
    # Check for STOP_LOSS_EXECUTED event
    executed_calls = [call for call in mock_event_bus.publish.call_args_list 
                     if call[0][0] == StopLossEvents.STOP_LOSS_EXECUTED]
    assert len(executed_calls) > 0

@pytest.mark.asyncio
async def test_portfolio_stop_loss(stop_loss_manager, mock_event_bus, mock_position_manager, mock_exchange_service):
    """Test portfolio percentage-based stop loss."""
    # Initialize the manager
    await stop_loss_manager.initialize()
    
    # Set initial portfolio value
    stop_loss_manager.stop_loss_metrics["initial_portfolio_value"] = 10000.0
    stop_loss_manager.stop_loss_metrics["max_portfolio_value"] = 12000.0
    
    # Mock current portfolio value (20% drawdown from max)
    mock_exchange_service.get_balance.return_value = {
        "USDT": {
            "free": 4000.0,
            "used": 5000.0,
            "total": 9000.0
        }
    }
    
    # Mock positions with negative PnL
    positions = [
        {
            "pair": "BTC/USDT",
            "position_side": "long",
            "size": 0.5,
            "entry_price": 20000.0,
            "unrealized_pnl": -400.0,
            "position_id": "BTCUSDT-long"
        }
    ]
    mock_position_manager.get_all_positions.return_value = positions
    
    # Call the portfolio stop loss monitoring method directly
    await stop_loss_manager._monitor_portfolio_stop_loss()
    
    # Verify that stop loss was executed (20% drawdown exceeds 10% threshold)
    mock_position_manager.close_position.assert_called_with(
        side="long",
        order_type="market"
    )

@pytest.mark.asyncio
async def test_trailing_stop_loss(stop_loss_manager, mock_event_bus, mock_position_manager, mock_exchange_service):
    """Test trailing stop loss."""
    # Initialize the manager
    await stop_loss_manager.initialize()
    
    # Enable trailing stop loss
    stop_loss_manager.stop_loss_settings["trailing_stop_loss"]["enabled"] = True
    stop_loss_manager.stop_loss_settings["trailing_stop_loss"]["activation_threshold"] = 0.02  # 2% profit
    stop_loss_manager.stop_loss_settings["trailing_stop_loss"]["trailing_distance"] = 0.01  # 1% trailing distance
    
    # Mock a position with profit
    position = {
        "pair": "BTC/USDT",
        "position_side": "long",
        "size": 1.0,
        "entry_price": 19500.0,  # Current price is 20000, so 2.56% profit
        "unrealized_pnl": 500.0,
        "position_id": "BTCUSDT-long"
    }
    
    # Set up position manager to return this position
    mock_position_manager.get_all_positions.return_value = [position]
    
    # First call to set up trailing stop
    await stop_loss_manager._monitor_trailing_stop_loss()
    
    # Verify trailing stop was set up
    assert len(stop_loss_manager._monitor_trailing_stop_loss.__closure__[0].cell_contents) == 1
    
    # Now simulate price dropping below trailing stop
    mock_exchange_service.get_current_price.return_value = 19800.0  # Drop below trailing stop
    
    # Second call to trigger trailing stop
    await stop_loss_manager._monitor_trailing_stop_loss()
    
    # Verify that stop loss was executed
    mock_position_manager.close_position.assert_called_with(
        side="long",
        order_type="market"
    )

@pytest.mark.asyncio
async def test_handle_position_events(stop_loss_manager):
    """Test handling of position events."""
    # Initialize the manager
    await stop_loss_manager.initialize()
    
    # Test position opened event
    position_data = {
        "pair": "BTC/USDT",
        "position_side": "long",
        "size": 1.0,
        "entry_price": 20000.0,
        "position_id": "BTCUSDT-long-new"
    }
    
    await stop_loss_manager._handle_position_opened({
        "position": position_data
    })
    
    # Verify position was added to tracking
    assert "BTCUSDT-long-new" in stop_loss_manager.stop_loss_metrics["positions_with_stop_loss"]
    
    # Test position closed event
    await stop_loss_manager._handle_position_closed({
        "position": position_data
    })
    
    # Verify position was removed from tracking
    assert "BTCUSDT-long-new" not in stop_loss_manager.stop_loss_metrics["positions_with_stop_loss"]

@pytest.mark.asyncio
async def test_update_stop_loss_settings(stop_loss_manager, mock_event_bus):
    """Test updating stop loss settings."""
    # Initialize the manager
    await stop_loss_manager.initialize()
    
    # Update settings
    new_settings = {
        "usdt_stop_loss": {
            "max_loss_amount": 1500.0,
            "warning_threshold": 0.8
        },
        "portfolio_stop_loss": {
            "max_loss_percentage": 0.15
        }
    }
    
    await stop_loss_manager.update_stop_loss_settings(new_settings)
    
    # Verify settings were updated
    assert stop_loss_manager.stop_loss_settings["usdt_stop_loss"]["max_loss_amount"] == 1500.0
    assert stop_loss_manager.stop_loss_settings["usdt_stop_loss"]["warning_threshold"] == 0.8
    assert stop_loss_manager.stop_loss_settings["portfolio_stop_loss"]["max_loss_percentage"] == 0.15
    
    # Verify event was published
    mock_event_bus.publish.assert_called_with(
        StopLossEvents.STOP_LOSS_SETTINGS_UPDATED,
        {
            "old_settings": stop_loss_manager.stop_loss_settings.copy(),
            "new_settings": stop_loss_manager.stop_loss_settings,
            "timestamp": mock_event_bus.publish.call_args[0][1]["timestamp"]
        }
    )

@pytest.mark.asyncio
async def test_shutdown(stop_loss_manager):
    """Test shutting down the stop loss manager."""
    # Initialize the manager
    await stop_loss_manager.initialize()
    
    # Create some mock tasks
    task1 = asyncio.create_task(asyncio.sleep(1))
    task2 = asyncio.create_task(asyncio.sleep(1))
    
    # Add tasks to the manager
    stop_loss_manager.stop_loss_update_tasks.add(task1)
    stop_loss_manager.stop_loss_update_tasks.add(task2)
    
    # Shutdown the manager
    await stop_loss_manager.shutdown()
    
    # Verify tasks were cancelled
    assert task1.cancelled()
    assert task2.cancelled()
    assert len(stop_loss_manager.stop_loss_update_tasks) == 0