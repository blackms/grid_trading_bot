import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from decimal import Decimal

from config.config_manager import ConfigManager
from core.services.exchange_interface import ExchangeInterface
from core.bot_management.event_bus import EventBus, Events
from core.order_handling.futures_position_manager import FuturesPositionManager, Position, PositionEvents

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
    return config_manager

@pytest.fixture
def mock_exchange_service():
    exchange_service = AsyncMock(spec=ExchangeInterface)
    exchange_service.set_leverage.return_value = {"leverage": 5, "marginType": "isolated"}
    exchange_service.get_positions.return_value = []
    exchange_service.get_current_price.return_value = 50000.0
    return exchange_service

@pytest.fixture
def mock_event_bus():
    event_bus = MagicMock(spec=EventBus)
    event_bus.publish = AsyncMock()
    return event_bus

@pytest.fixture
async def position_manager(mock_config_manager, mock_exchange_service, mock_event_bus):
    manager = FuturesPositionManager(
        config_manager=mock_config_manager,
        exchange_service=mock_exchange_service,
        event_bus=mock_event_bus
    )
    return manager

@pytest.mark.asyncio
async def test_initialization(position_manager, mock_exchange_service):
    # Test initialization
    await position_manager.initialize()
    
    # Verify leverage was set
    mock_exchange_service.set_leverage.assert_called_once_with(
        pair="BTCUSDT",
        leverage=5,
        margin_mode="isolated"
    )
    
    # Verify positions were loaded
    mock_exchange_service.get_positions.assert_called_with("BTCUSDT")

@pytest.mark.asyncio
async def test_open_position(position_manager, mock_exchange_service):
    # Mock the place_order response
    mock_exchange_service.place_order.return_value = {
        "orderId": "12345",
        "status": "FILLED",
        "executedQty": 1.0,
        "price": 50000.0
    }
    
    # Mock the position data that will be returned after placing the order
    position_data = {
        "symbol": "BTCUSDT",
        "side": "long",
        "size": 1.0,
        "entryPrice": 50000.0,
        "leverage": 5,
        "marginType": "isolated",
        "initialMargin": 10000.0,
        "liquidationPrice": 40000.0,
        "unrealizedPnl": 0.0,
        "realizedPnl": 0.0
    }
    mock_exchange_service.get_positions.return_value = [position_data]
    
    # Open a position
    result = await position_manager.open_position(
        side="long",
        size=1.0,
        price=50000.0,
        order_type="limit"
    )
    
    # Verify the order was placed
    mock_exchange_service.place_order.assert_called_once_with(
        pair="BTCUSDT",
        order_side="buy",
        order_type="limit",
        amount=1.0,
        price=50000.0
    )
    
    # Verify positions were reloaded
    assert mock_exchange_service.get_positions.call_count >= 2
    
    # Verify the result
    assert result["success"] is True
    assert result["position"]["pair"] == "BTCUSDT"
    assert result["position"]["position_side"] == "long"
    assert result["position"]["size"] == 1.0
    assert result["position"]["entry_price"] == 50000.0

@pytest.mark.asyncio
async def test_modify_position(position_manager, mock_exchange_service, mock_event_bus):
    # Set up an existing position
    position_data = {
        "symbol": "BTCUSDT",
        "side": "long",
        "size": 1.0,
        "entryPrice": 50000.0,
        "leverage": 5,
        "marginType": "isolated",
        "initialMargin": 10000.0,
        "liquidationPrice": 40000.0,
        "unrealizedPnl": 0.0,
        "realizedPnl": 0.0
    }
    mock_exchange_service.get_positions.return_value = [position_data]
    await position_manager.load_positions()
    
    # Mock the place_order response for modification
    mock_exchange_service.place_order.return_value = {
        "orderId": "12346",
        "status": "FILLED",
        "executedQty": 0.5,
        "price": 51000.0
    }
    
    # Mock the updated position data
    updated_position_data = {
        "symbol": "BTCUSDT",
        "side": "long",
        "size": 1.5,  # Increased by 0.5
        "entryPrice": 50333.33,  # Weighted average of old and new prices
        "leverage": 5,
        "marginType": "isolated",
        "initialMargin": 15100.0,
        "liquidationPrice": 40266.67,
        "unrealizedPnl": 1000.0,  # Some profit due to price increase
        "realizedPnl": 0.0
    }
    mock_exchange_service.get_positions.return_value = [updated_position_data]
    
    # Modify the position (increase size)
    result = await position_manager.modify_position(
        side="long",
        size_change=0.5,
        price=51000.0,
        order_type="limit"
    )
    
    # Verify the order was placed
    mock_exchange_service.place_order.assert_called_once_with(
        pair="BTCUSDT",
        order_side="buy",
        order_type="limit",
        amount=0.5,
        price=51000.0
    )
    
    # Verify positions were reloaded
    assert mock_exchange_service.get_positions.call_count >= 2
    
    # Verify the event was published
    mock_event_bus.publish.assert_called_with(
        PositionEvents.POSITION_MODIFIED,
        {
            "pair": "BTCUSDT",
            "position_side": "long",
            "old_size": 1.0,
            "new_size": 1.5,
            "size_change": 0.5,
            "entry_price": 50333.33,
            "position": position_manager.positions["BTCUSDT-long"].to_dict(),
            "order": mock_exchange_service.place_order.return_value
        }
    )
    
    # Verify the result
    assert result["success"] is True
    assert result["position"]["size"] == 1.5
    assert result["position"]["entry_price"] == 50333.33

@pytest.mark.asyncio
async def test_close_position(position_manager, mock_exchange_service, mock_event_bus):
    # Set up an existing position
    position_data = {
        "symbol": "BTCUSDT",
        "side": "long",
        "size": 1.0,
        "entryPrice": 50000.0,
        "leverage": 5,
        "marginType": "isolated",
        "initialMargin": 10000.0,
        "liquidationPrice": 40000.0,
        "unrealizedPnl": 1000.0,
        "realizedPnl": 0.0
    }
    mock_exchange_service.get_positions.return_value = [position_data]
    await position_manager.load_positions()
    
    # Mock the close_position response
    mock_exchange_service.close_position.return_value = {
        "success": True,
        "orderId": "12347",
        "status": "FILLED",
        "executedQty": 1.0,
        "price": 51000.0,
        "realizedPnl": 1000.0
    }
    
    # After closing, no positions should be returned
    mock_exchange_service.get_positions.return_value = []
    
    # Close the position
    result = await position_manager.close_position(side="long")
    
    # Verify close_position was called
    mock_exchange_service.close_position.assert_called_once_with(
        pair="BTCUSDT",
        position_side="long"
    )
    
    # Verify the event was published
    mock_event_bus.publish.assert_called_with(
        PositionEvents.POSITION_CLOSED,
        {
            "pair": "BTCUSDT",
            "position_side": "long",
            "size": 1.0,
            "realized_pnl": 0.0,  # From the original position data
            "position": position_manager.positions.get("BTCUSDT-long", {"pair": "BTCUSDT", "position_side": "long", "size": 1.0, "realized_pnl": 0.0}).to_dict(),
            "result": mock_exchange_service.close_position.return_value
        }
    )
    
    # Verify the result
    assert result["success"] is True
    assert result["position_closed"] is True

@pytest.mark.asyncio
async def test_partial_close_position(position_manager, mock_exchange_service, mock_event_bus):
    # Set up an existing position
    position_data = {
        "symbol": "BTCUSDT",
        "side": "long",
        "size": 2.0,
        "entryPrice": 50000.0,
        "leverage": 5,
        "marginType": "isolated",
        "initialMargin": 20000.0,
        "liquidationPrice": 40000.0,
        "unrealizedPnl": 2000.0,
        "realizedPnl": 0.0
    }
    mock_exchange_service.get_positions.return_value = [position_data]
    await position_manager.load_positions()
    
    # Mock the place_order response for partial close
    mock_exchange_service.place_order.return_value = {
        "orderId": "12348",
        "status": "FILLED",
        "executedQty": 1.0,
        "price": 51000.0,
        "realizedPnl": 1000.0
    }
    
    # Mock the updated position data after partial close
    updated_position_data = {
        "symbol": "BTCUSDT",
        "side": "long",
        "size": 1.0,  # Reduced by 1.0
        "entryPrice": 50000.0,  # Same entry price
        "leverage": 5,
        "marginType": "isolated",
        "initialMargin": 10000.0,
        "liquidationPrice": 40000.0,
        "unrealizedPnl": 1000.0,
        "realizedPnl": 1000.0  # Realized some profit
    }
    mock_exchange_service.get_positions.return_value = [updated_position_data]
    
    # Partially close the position (50%)
    result = await position_manager.close_position(
        side="long",
        percentage=50.0,
        price=51000.0,
        order_type="limit"
    )
    
    # Verify the order was placed with reduce-only flag
    mock_exchange_service.place_order.assert_called_once_with(
        pair="BTCUSDT",
        order_side="sell",
        order_type="limit",
        amount=1.0,  # 50% of 2.0
        price=51000.0,
        params={"reduceOnly": True}
    )
    
    # Verify positions were reloaded
    assert mock_exchange_service.get_positions.call_count >= 2
    
    # Verify the event was published
    mock_event_bus.publish.assert_called_with(
        PositionEvents.POSITION_MODIFIED,
        {
            "pair": "BTCUSDT",
            "position_side": "long",
            "old_size": 2.0,
            "new_size": 1.0,
            "size_change": -1.0,
            "entry_price": 50000.0,
            "position": position_manager.positions["BTCUSDT-long"].to_dict(),
            "order": mock_exchange_service.place_order.return_value
        }
    )
    
    # Verify the result
    assert result["success"] is True
    assert result["position_closed"] is False
    assert result["position"]["size"] == 1.0
    assert result["position"]["realized_pnl"] == 1000.0

@pytest.mark.asyncio
async def test_liquidation_risk_warning(position_manager, mock_exchange_service, mock_event_bus):
    # Set up a position close to liquidation
    position_data = {
        "symbol": "BTCUSDT",
        "side": "long",
        "size": 1.0,
        "entryPrice": 50000.0,
        "leverage": 5,
        "marginType": "isolated",
        "initialMargin": 10000.0,
        "liquidationPrice": 45000.0,  # Close to current price
        "unrealizedPnl": -4500.0,  # Losing position
        "realizedPnl": 0.0
    }
    mock_exchange_service.get_positions.return_value = [position_data]
    await position_manager.load_positions()
    
    # Set current price close to liquidation price
    mock_exchange_service.get_current_price.return_value = 45500.0  # Only 1% away from liquidation
    
    # Manually trigger liquidation check
    await position_manager._check_liquidation_risk(position_manager.positions["BTCUSDT-long"])
    
    # Verify the warning event was published
    mock_event_bus.publish.assert_called_with(
        PositionEvents.POSITION_LIQUIDATION_WARNING,
        {
            "pair": "BTCUSDT",
            "position_side": "long",
            "current_price": 45500.0,
            "liquidation_price": 45000.0,
            "distance_to_liquidation": (45500.0 - 45000.0) / 45500.0,  # About 1.1%
            "threshold": 0.1,
            "position": position_manager.positions["BTCUSDT-long"].to_dict()
        }
    )

@pytest.mark.asyncio
async def test_position_tracking(position_manager, mock_exchange_service, mock_event_bus):
    # Set up initial position
    position_data = {
        "symbol": "BTCUSDT",
        "side": "long",
        "size": 1.0,
        "entryPrice": 50000.0,
        "leverage": 5,
        "marginType": "isolated",
        "initialMargin": 10000.0,
        "liquidationPrice": 40000.0,
        "unrealizedPnl": 0.0,
        "realizedPnl": 0.0
    }
    mock_exchange_service.get_positions.return_value = [position_data]
    await position_manager.load_positions()
    
    # Mock updated position with significant PnL change
    updated_position_data = {
        "symbol": "BTCUSDT",
        "side": "long",
        "size": 1.0,
        "entryPrice": 50000.0,
        "leverage": 5,
        "marginType": "isolated",
        "initialMargin": 10000.0,
        "liquidationPrice": 40000.0,
        "unrealizedPnl": 1000.0,  # Significant change
        "realizedPnl": 0.0
    }
    mock_exchange_service.get_positions.return_value = [updated_position_data]
    
    # Manually trigger position update
    await position_manager._update_positions()
    
    # Verify PnL update event was published
    mock_event_bus.publish.assert_called_with(
        PositionEvents.POSITION_PNL_UPDATE,
        {
            "pair": "BTCUSDT",
            "position_side": "long",
            "unrealized_pnl": 1000.0,
            "realized_pnl": 0.0,
            "entry_price": 50000.0,
            "current_price": position_manager._calculate_current_price(position_manager.positions["BTCUSDT-long"]),
            "position": position_manager.positions["BTCUSDT-long"].to_dict()
        }
    )

@pytest.mark.asyncio
async def test_shutdown(position_manager):
    # Initialize the manager to create tasks
    await position_manager.initialize()
    
    # Verify tasks were created
    assert len(position_manager.position_update_tasks) > 0
    
    # Shutdown the manager
    await position_manager.shutdown()
    
    # Verify tasks were cancelled and cleared
    assert len(position_manager.position_update_tasks) == 0