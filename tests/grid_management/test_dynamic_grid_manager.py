import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
import numpy as np

from config.config_manager import ConfigManager
from strategies.strategy_type import StrategyType
from strategies.spacing_type import SpacingType
from core.grid_management.dynamic_grid_manager import DynamicGridManager
from core.grid_management.grid_level import GridLevel, GridCycleState
from core.order_handling.order import Order, OrderSide
from core.order_handling.futures_position_manager import FuturesPositionManager
from core.order_handling.futures_risk_manager import FuturesRiskManager
from core.bot_management.event_bus import EventBus

@pytest.fixture
def mock_config_manager():
    config_manager = MagicMock(spec=ConfigManager)
    
    # Mock grid settings
    grid_settings = {
        "type": "simple_grid",
        "spacing": "arithmetic",
        "num_grids": 5,
        "range": {
            "top": 22000.0,
            "bottom": 18000.0
        },
        "dynamic_grid": {
            "trailing_enabled": True,
            "trailing_activation_threshold": 0.02,
            "trailing_distance_percentage": 0.01,
            "trailing_cooldown_period": 300,
            "volatility_adaptation_enabled": True,
            "volatility_lookback_period": 24,
            "volatility_grid_adjustment_factor": 1.5,
            "grid_repositioning_enabled": True,
            "grid_repositioning_threshold": 0.05,
            "small_capital_optimization_enabled": True,
            "min_order_value": 5.0
        }
    }
    
    # Mock config methods
    config_manager.get_grid_settings.return_value = grid_settings
    config_manager.get_num_grids.return_value = 5
    config_manager.get_spacing_type.return_value = SpacingType.ARITHMETIC
    config_manager.get_top_range.return_value = 22000.0
    config_manager.get_bottom_range.return_value = 18000.0
    config_manager.get_leverage.return_value = 3
    
    return config_manager

@pytest.fixture
def mock_position_manager():
    position_manager = AsyncMock(spec=FuturesPositionManager)
    
    # Mock position data
    position_data = {
        'pair': 'BTC/USDT',
        'position_side': 'long',
        'size': 0.1,
        'entry_price': 20000.0,
        'leverage': 3,
        'margin_type': 'isolated',
        'initial_margin': 666.67,
        'liquidation_price': 18000.0,
        'unrealized_pnl': 0.0,
        'realized_pnl': 0.0,
        'position_id': '12345'
    }
    
    # Mock methods
    position_manager.get_all_positions.return_value = [position_data]
    position_manager.get_position.return_value = position_data
    
    return position_manager

@pytest.fixture
def mock_risk_manager():
    risk_manager = AsyncMock(spec=FuturesRiskManager)
    
    # Mock risk data
    risk_info = {
        "max_size": 0.5,
        "within_limits": True,
        "risk_factors": {
            "leverage": 3,
            "liquidation_distance": 0.1,
            "margin_health": 0.8
        }
    }
    
    # Mock methods
    risk_manager.is_position_within_risk_limits.return_value = (True, risk_info)
    risk_manager.calculate_max_position_size.return_value = 0.5
    risk_manager.circuit_breaker_active = False
    
    return risk_manager

@pytest.fixture
def mock_event_bus():
    event_bus = AsyncMock(spec=EventBus)
    return event_bus

@pytest.fixture
def dynamic_grid_manager(mock_config_manager, mock_position_manager, mock_risk_manager, mock_event_bus):
    return DynamicGridManager(
        config_manager=mock_config_manager,
        strategy_type=StrategyType.SIMPLE_GRID,
        position_manager=mock_position_manager,
        risk_manager=mock_risk_manager,
        event_bus=mock_event_bus
    )

class TestDynamicGridManager:
    
    @pytest.mark.asyncio
    async def test_initialization(self, dynamic_grid_manager):
        """Test that the DynamicGridManager initializes correctly."""
        # Check that dynamic grid settings are loaded
        assert dynamic_grid_manager.trailing_enabled is True
        assert dynamic_grid_manager.trailing_activation_threshold == 0.02
        assert dynamic_grid_manager.trailing_distance_percentage == 0.01
        assert dynamic_grid_manager.trailing_cooldown_period == 300
        
        # Initialize dynamic grids
        await dynamic_grid_manager.initialize_dynamic_grids()
        
        # Check that grids are initialized
        assert len(dynamic_grid_manager.price_grids) == 5
        assert dynamic_grid_manager.central_price == 20000.0
        assert dynamic_grid_manager.trailing_reference_price == 20000.0
    
    @pytest.mark.asyncio
    async def test_get_order_size_with_leverage(self, dynamic_grid_manager):
        """Test that order size calculation accounts for leverage."""
        # Mock the parent method
        with patch.object(dynamic_grid_manager, 'get_order_size_for_grid_level', return_value=0.1):
            # Calculate order size with leverage
            order_size = await dynamic_grid_manager.get_order_size_for_grid_level_with_leverage(10000.0, 20000.0)
            
            # Check that leverage is applied (0.1 * 3 = 0.3)
            assert order_size == 0.3
    
    @pytest.mark.asyncio
    async def test_small_capital_optimization(self, dynamic_grid_manager):
        """Test that small capital optimization works correctly."""
        # Set a very small base order size
        with patch.object(dynamic_grid_manager, 'get_order_size_for_grid_level', return_value=0.0001):
            # Calculate order size with small capital optimization
            order_size = await dynamic_grid_manager.get_order_size_for_grid_level_with_leverage(100.0, 20000.0)
            
            # Check that minimum order value is enforced (5.0 / 20000.0 = 0.00025)
            assert order_size == 0.00025
    
    @pytest.mark.asyncio
    async def test_risk_limit_enforcement(self, dynamic_grid_manager, mock_risk_manager):
        """Test that risk limits are enforced for order sizes."""
        # Mock the parent method to return a large order size
        with patch.object(dynamic_grid_manager, 'get_order_size_for_grid_level', return_value=0.5):
            # Mock risk manager to indicate the order exceeds limits
            mock_risk_manager.is_position_within_risk_limits.return_value = (False, {"max_size": 0.3})
            
            # Calculate order size with risk limits
            order_size = await dynamic_grid_manager.get_order_size_for_grid_level_with_leverage(10000.0, 20000.0)
            
            # Check that order size is limited to max allowed
            assert order_size == 0.3
    
    @pytest.mark.asyncio
    async def test_trailing_grid_adjustment_up(self, dynamic_grid_manager):
        """Test that grids adjust correctly when price moves up significantly."""
        # Initialize dynamic grids
        await dynamic_grid_manager.initialize_dynamic_grids()
        
        # Set initial state
        dynamic_grid_manager.trailing_reference_price = 20000.0
        dynamic_grid_manager.trailing_direction = None
        dynamic_grid_manager.last_grid_adjustment_time = 0
        
        # Mock current price to be significantly higher
        with patch.object(dynamic_grid_manager, '_get_current_price', return_value=21000.0):
            # Check trailing conditions
            await dynamic_grid_manager._check_trailing_conditions(21000.0)
            
            # First call should just set the direction
            assert dynamic_grid_manager.trailing_direction == "up"
            assert dynamic_grid_manager.trailing_reference_price == 21000.0
            
            # Mock another price increase
            with patch.object(dynamic_grid_manager, '_get_current_price', return_value=22000.0):
                # Check trailing conditions again
                await dynamic_grid_manager._check_trailing_conditions(22000.0)
                
                # Now grids should adjust
                assert dynamic_grid_manager.central_price > 20000.0
                assert dynamic_grid_manager.trailing_reference_price == 22000.0
                
                # Event should be published
                dynamic_grid_manager.event_bus.publish.assert_called_with(
                    "grid_adjustment",
                    {
                        "old_center": 20000.0,
                        "new_center": dynamic_grid_manager.central_price,
                        "adjustment_type": "up",
                        "price_grids": dynamic_grid_manager.price_grids,
                        "timestamp": pytest.approx(dynamic_grid_manager.last_grid_adjustment_time, abs=1)
                    }
                )
    
    @pytest.mark.asyncio
    async def test_trailing_grid_adjustment_down(self, dynamic_grid_manager):
        """Test that grids adjust correctly when price moves down significantly."""
        # Initialize dynamic grids
        await dynamic_grid_manager.initialize_dynamic_grids()
        
        # Set initial state
        dynamic_grid_manager.trailing_reference_price = 20000.0
        dynamic_grid_manager.trailing_direction = None
        dynamic_grid_manager.last_grid_adjustment_time = 0
        
        # Mock current price to be significantly lower
        with patch.object(dynamic_grid_manager, '_get_current_price', return_value=19000.0):
            # Check trailing conditions
            await dynamic_grid_manager._check_trailing_conditions(19000.0)
            
            # First call should just set the direction
            assert dynamic_grid_manager.trailing_direction == "down"
            assert dynamic_grid_manager.trailing_reference_price == 19000.0
            
            # Mock another price decrease
            with patch.object(dynamic_grid_manager, '_get_current_price', return_value=18000.0):
                # Check trailing conditions again
                await dynamic_grid_manager._check_trailing_conditions(18000.0)
                
                # Now grids should adjust
                assert dynamic_grid_manager.central_price < 20000.0
                assert dynamic_grid_manager.trailing_reference_price == 18000.0
                
                # Event should be published
                dynamic_grid_manager.event_bus.publish.assert_called_with(
                    "grid_adjustment",
                    {
                        "old_center": 20000.0,
                        "new_center": dynamic_grid_manager.central_price,
                        "adjustment_type": "down",
                        "price_grids": dynamic_grid_manager.price_grids,
                        "timestamp": pytest.approx(dynamic_grid_manager.last_grid_adjustment_time, abs=1)
                    }
                )
    
    @pytest.mark.asyncio
    async def test_volatility_adaptation(self, dynamic_grid_manager):
        """Test that grid spacing adapts to market volatility."""
        # Initialize dynamic grids
        await dynamic_grid_manager.initialize_dynamic_grids()
        
        # Set initial price history with low volatility
        dynamic_grid_manager.price_history = [20000.0 * (1 + 0.0001 * i) for i in range(24)]
        
        # Add high volatility data
        dynamic_grid_manager.price_history.extend([20000.0 * (1 + 0.01 * i) for i in range(12)])
        
        # Check volatility adaptation
        await dynamic_grid_manager._check_volatility_adaptation()
        
        # Volatility should be detected as high
        assert len(dynamic_grid_manager.volatility_history) > 0
        
        # Event should be published
        dynamic_grid_manager.event_bus.publish.assert_called_with(
            "grid_adaptation",
            {
                "current_volatility": pytest.approx(dynamic_grid_manager.volatility_history[-1], rel=0.1),
                "avg_volatility": pytest.approx(np.mean(dynamic_grid_manager.volatility_history), rel=0.1),
                "adaptation_type": "widen",
                "price_grids": dynamic_grid_manager.price_grids,
                "timestamp": pytest.approx(dynamic_grid_manager.last_grid_adjustment_time, abs=1)
            }
        )
    
    @pytest.mark.asyncio
    async def test_grid_repositioning(self, dynamic_grid_manager):
        """Test that grids reposition when price moves far from central price."""
        # Initialize dynamic grids
        await dynamic_grid_manager.initialize_dynamic_grids()
        
        # Set initial state
        dynamic_grid_manager.central_price = 20000.0
        dynamic_grid_manager.last_grid_adjustment_time = 0
        
        # Mock current price to be far from central price
        current_price = 22000.0  # 10% higher than central price
        
        # Check grid repositioning
        await dynamic_grid_manager._check_grid_repositioning(current_price)
        
        # Grids should be repositioned
        assert dynamic_grid_manager.central_price == current_price
        
        # Event should be published
        dynamic_grid_manager.event_bus.publish.assert_called_with(
            "grid_adjustment",
            {
                "old_center": 20000.0,
                "new_center": current_price,
                "adjustment_type": "reposition",
                "price_grids": dynamic_grid_manager.price_grids,
                "timestamp": pytest.approx(dynamic_grid_manager.last_grid_adjustment_time, abs=1)
            }
        )
    
    @pytest.mark.asyncio
    async def test_shutdown(self, dynamic_grid_manager):
        """Test that shutdown cancels all background tasks."""
        # Create a mock task
        mock_task = MagicMock()
        mock_task.cancel = MagicMock()
        
        # Add task to grid adjustment tasks
        dynamic_grid_manager.grid_adjustment_tasks.add(mock_task)
        
        # Shutdown
        await dynamic_grid_manager.shutdown()
        
        # Task should be cancelled
        mock_task.cancel.assert_called_once()