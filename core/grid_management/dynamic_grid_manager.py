import logging
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
import asyncio
import time

from config.config_manager import ConfigManager
from strategies.strategy_type import StrategyType
from strategies.spacing_type import SpacingType
from .grid_manager import GridManager
from .grid_level import GridLevel, GridCycleState
from ..order_handling.order import Order, OrderSide
from ..order_handling.futures_position_manager import FuturesPositionManager
from ..order_handling.futures_risk_manager import FuturesRiskManager
from core.bot_management.event_bus import EventBus

class DynamicGridManager(GridManager):
    """
    DynamicGridManager extends the GridManager to support dynamic grid adjustments
    based on price movements with trailing up/down functionality.
    
    Key features:
    1. Dynamic grid adjustment based on price movements
    2. Trailing up/down functionality
    3. Grid repositioning coordination
    4. Support for small capital with high leverage
    5. Adaptation to market volatility
    """
    
    def __init__(
        self, 
        config_manager: ConfigManager, 
        strategy_type: StrategyType,
        position_manager: FuturesPositionManager,
        risk_manager: FuturesRiskManager,
        event_bus: EventBus
    ):
        """
        Initialize the DynamicGridManager.
        
        Args:
            config_manager: Configuration manager instance
            strategy_type: The grid trading strategy type
            position_manager: Futures position manager for position tracking
            risk_manager: Futures risk manager for risk assessment
            event_bus: Event bus for publishing grid events
        """
        super().__init__(config_manager, strategy_type)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.position_manager = position_manager
        self.risk_manager = risk_manager
        self.event_bus = event_bus
        
        # Dynamic grid settings
        self.dynamic_grid_settings = self._load_dynamic_grid_settings()
        
        # Trailing settings
        self.trailing_enabled = self.dynamic_grid_settings.get("trailing_enabled", True)
        self.trailing_activation_threshold = self.dynamic_grid_settings.get("trailing_activation_threshold", 0.02)  # 2% price movement
        self.trailing_distance_percentage = self.dynamic_grid_settings.get("trailing_distance_percentage", 0.01)  # 1% trailing distance
        self.trailing_cooldown_period = self.dynamic_grid_settings.get("trailing_cooldown_period", 300)  # 5 minutes
        
        # Volatility adaptation settings
        self.volatility_adaptation_enabled = self.dynamic_grid_settings.get("volatility_adaptation_enabled", True)
        self.volatility_lookback_period = self.dynamic_grid_settings.get("volatility_lookback_period", 24)  # Hours
        self.volatility_grid_adjustment_factor = self.dynamic_grid_settings.get("volatility_grid_adjustment_factor", 1.5)
        
        # Grid repositioning settings
        self.grid_repositioning_enabled = self.dynamic_grid_settings.get("grid_repositioning_enabled", True)
        self.grid_repositioning_threshold = self.dynamic_grid_settings.get("grid_repositioning_threshold", 0.05)  # 5% from central price
        
        # Small capital optimization settings
        self.small_capital_optimization_enabled = self.dynamic_grid_settings.get("small_capital_optimization_enabled", True)
        self.min_order_value = self.dynamic_grid_settings.get("min_order_value", 5.0)  # Minimum order value in quote currency
        
        # State tracking
        self.last_grid_adjustment_time = 0
        self.trailing_reference_price = 0
        self.trailing_direction = None  # "up" or "down"
        self.price_history = []
        self.volatility_history = []
        self.is_grid_adjustment_active = False
        self.grid_adjustment_tasks = set()
        
        self.logger.info(f"Initialized DynamicGridManager with trailing {'enabled' if self.trailing_enabled else 'disabled'}")
    
    def _load_dynamic_grid_settings(self) -> Dict[str, Any]:
        """
        Load dynamic grid settings from configuration.
        
        Returns:
            Dictionary containing dynamic grid settings
        """
        # Get dynamic grid settings from config or use defaults
        grid_settings = self.config_manager.get_grid_settings()
        return grid_settings.get("dynamic_grid", {})
    
    async def initialize_dynamic_grids(self) -> None:
        """
        Initialize dynamic grids and start monitoring tasks.
        """
        # Initialize standard grids first
        self.initialize_grids_and_levels()
        
        # Set initial trailing reference price to central price
        self.trailing_reference_price = self.central_price
        
        # Start grid adjustment monitoring task
        if self.trailing_enabled or self.volatility_adaptation_enabled:
            self._start_grid_adjustment_monitoring()
        
        self.logger.info(f"Dynamic grids initialized with central price: {self.central_price}")
    
    def _start_grid_adjustment_monitoring(self) -> None:
        """
        Start background task for monitoring price movements and adjusting grids.
        """
        adjustment_task = asyncio.create_task(self._monitor_for_grid_adjustments())
        self.grid_adjustment_tasks.add(adjustment_task)
        adjustment_task.add_done_callback(self.grid_adjustment_tasks.discard)
        
        self.logger.info("Started grid adjustment monitoring task")
    
    async def _monitor_for_grid_adjustments(self) -> None:
        """
        Monitor price movements and trigger grid adjustments when necessary.
        """
        try:
            while True:
                # Skip if circuit breaker is active
                if hasattr(self.risk_manager, 'circuit_breaker_active') and self.risk_manager.circuit_breaker_active:
                    await asyncio.sleep(10)
                    continue
                
                # Get current price
                current_price = await self._get_current_price()
                
                # Add to price history
                self.price_history.append(current_price)
                if len(self.price_history) > self.volatility_lookback_period * 6:  # Keep 6 data points per hour
                    self.price_history.pop(0)
                
                # Check for trailing conditions
                if self.trailing_enabled:
                    await self._check_trailing_conditions(current_price)
                
                # Check for volatility adaptation
                if self.volatility_adaptation_enabled and len(self.price_history) >= 12:  # Need at least 2 hours of data
                    await self._check_volatility_adaptation()
                
                # Check for grid repositioning
                if self.grid_repositioning_enabled:
                    await self._check_grid_repositioning(current_price)
                
                # Sleep for 10 seconds before next check
                await asyncio.sleep(10)
        except asyncio.CancelledError:
            self.logger.info("Grid adjustment monitoring task cancelled")
        except Exception as e:
            self.logger.error(f"Error in grid adjustment monitoring: {e}")
    
    async def _get_current_price(self) -> float:
        """
        Get the current price from the exchange.
        
        Returns:
            Current price as a float
        """
        # In a real implementation, this would get the price from the exchange service
        # For now, we'll use a simple approach to get the price from the position manager
        
        # Try to get the price from an existing position
        positions = await self.position_manager.get_all_positions()
        if positions:
            # Use the first position's entry price as an approximation
            position = positions[0]
            return position.get('entry_price', self.central_price)
        
        # If no positions, use the central price
        return self.central_price
    
    async def _check_trailing_conditions(self, current_price: float) -> None:
        """
        Check if trailing conditions are met and adjust grids accordingly.
        
        Args:
            current_price: The current market price
        """
        # Skip if we're in cooldown period
        current_time = time.time()
        if current_time - self.last_grid_adjustment_time < self.trailing_cooldown_period:
            return
        
        # Calculate price movement percentage
        price_movement = (current_price - self.trailing_reference_price) / self.trailing_reference_price
        
        # Check if price movement exceeds threshold
        if abs(price_movement) >= self.trailing_activation_threshold:
            # Determine trailing direction
            new_direction = "up" if price_movement > 0 else "down"
            
            # If direction changed, update reference price
            if self.trailing_direction != new_direction:
                self.trailing_direction = new_direction
                self.trailing_reference_price = current_price
                self.logger.info(f"Trailing direction changed to {new_direction} at price {current_price}")
                return
            
            # Calculate trailing distance
            trailing_distance = current_price * self.trailing_distance_percentage
            
            # Calculate new grid center based on trailing direction
            if self.trailing_direction == "up":
                # For upward trend, trail behind the price
                new_center = current_price - trailing_distance
                if new_center > self.central_price:
                    await self._adjust_grids(new_center, "up")
            else:
                # For downward trend, trail above the price
                new_center = current_price + trailing_distance
                if new_center < self.central_price:
                    await self._adjust_grids(new_center, "down")
    
    async def _check_volatility_adaptation(self) -> None:
        """
        Check market volatility and adapt grid spacing accordingly.
        """
        # Calculate current volatility (standard deviation of returns)
        returns = [self.price_history[i] / self.price_history[i-1] - 1 for i in range(1, len(self.price_history))]
        current_volatility = np.std(returns) * 100  # Convert to percentage
        
        # Add to volatility history
        self.volatility_history.append(current_volatility)
        if len(self.volatility_history) > 24:  # Keep 24 hours of volatility data
            self.volatility_history.pop(0)
        
        # Calculate average volatility
        avg_volatility = np.mean(self.volatility_history) if self.volatility_history else current_volatility
        
        # Check if current volatility is significantly different from average
        if current_volatility > avg_volatility * self.volatility_grid_adjustment_factor:
            # High volatility - widen grid spacing
            await self._adapt_grid_spacing(current_volatility, avg_volatility, "widen")
        elif current_volatility < avg_volatility / self.volatility_grid_adjustment_factor:
            # Low volatility - narrow grid spacing
            await self._adapt_grid_spacing(current_volatility, avg_volatility, "narrow")
    
    async def _check_grid_repositioning(self, current_price: float) -> None:
        """
        Check if grid repositioning is needed based on price movement.
        
        Args:
            current_price: The current market price
        """
        # Calculate distance from central price
        distance_from_center = abs(current_price - self.central_price) / self.central_price
        
        # Check if price has moved significantly from central price
        if distance_from_center >= self.grid_repositioning_threshold:
            # Reposition grids around current price
            await self._adjust_grids(current_price, "reposition")
    
    async def _adjust_grids(self, new_center: float, adjustment_type: str) -> None:
        """
        Adjust grid levels based on new center price.
        
        Args:
            new_center: The new central price for the grid
            adjustment_type: Type of adjustment ("up", "down", "reposition")
        """
        if self.is_grid_adjustment_active:
            return
        
        try:
            self.is_grid_adjustment_active = True
            
            # Store old grid levels for reference
            old_grid_levels = self.grid_levels.copy()
            old_central_price = self.central_price
            
            # Update central price
            self.central_price = new_center
            
            # Recalculate price grids
            self.price_grids, _ = self._calculate_price_grids_and_central_price(new_center)
            
            # Update sorted buy/sell grids
            if self.strategy_type == StrategyType.SIMPLE_GRID:
                self.sorted_buy_grids = [price_grid for price_grid in self.price_grids if price_grid <= self.central_price]
                self.sorted_sell_grids = [price_grid for price_grid in self.price_grids if price_grid > self.central_price]
            elif self.strategy_type == StrategyType.HEDGED_GRID:
                self.sorted_buy_grids = self.price_grids[:-1]  # All except the top grid
                self.sorted_sell_grids = self.price_grids[1:]  # All except the bottom grid
            
            # Create new grid levels with states transferred from old levels where possible
            new_grid_levels = {}
            for price in self.price_grids:
                # Find closest old grid level
                closest_old_price = min(old_grid_levels.keys(), key=lambda x: abs(x - price), default=None)
                
                if closest_old_price is not None and abs(closest_old_price - price) / price < 0.02:  # Within 2%
                    # Transfer state from closest old grid level
                    old_level = old_grid_levels[closest_old_price]
                    new_grid_levels[price] = GridLevel(price, old_level.state)
                    new_grid_levels[price].orders = old_level.orders
                    new_grid_levels[price].paired_buy_level = None  # Will be updated later
                    new_grid_levels[price].paired_sell_level = None  # Will be updated later
                else:
                    # Create new grid level with default state
                    if self.strategy_type == StrategyType.SIMPLE_GRID:
                        state = GridCycleState.READY_TO_BUY if price <= self.central_price else GridCycleState.READY_TO_SELL
                    else:  # HEDGED_GRID
                        state = GridCycleState.READY_TO_BUY_OR_SELL if price != self.price_grids[-1] else GridCycleState.READY_TO_SELL
                    new_grid_levels[price] = GridLevel(price, state)
            
            # Update grid levels
            self.grid_levels = new_grid_levels
            
            # Update trailing reference price
            self.trailing_reference_price = new_center
            
            # Update last adjustment time
            self.last_grid_adjustment_time = time.time()
            
            # Log the adjustment
            self.logger.info(f"Grid adjusted: {adjustment_type} from {old_central_price} to {new_center}")
            self.logger.info(f"New price grids: {self.price_grids}")
            self.logger.info(f"New buy grids: {self.sorted_buy_grids}")
            self.logger.info(f"New sell grids: {self.sorted_sell_grids}")
            
            # Publish grid adjustment event
            await self._publish_grid_adjustment_event(old_central_price, new_center, adjustment_type)
            
        finally:
            self.is_grid_adjustment_active = False
    
    async def _adapt_grid_spacing(self, current_volatility: float, avg_volatility: float, adaptation_type: str) -> None:
        """
        Adapt grid spacing based on market volatility.
        
        Args:
            current_volatility: Current market volatility
            avg_volatility: Average market volatility
            adaptation_type: Type of adaptation ("widen" or "narrow")
        """
        if self.is_grid_adjustment_active:
            return
        
        try:
            self.is_grid_adjustment_active = True
            
            # Get current grid settings
            grid_settings = self.config_manager.get_grid_settings()
            current_range = grid_settings.get('range', {})
            current_top = current_range.get('top', 0)
            current_bottom = current_range.get('bottom', 0)
            
            # Calculate new range based on volatility
            volatility_ratio = current_volatility / avg_volatility
            
            if adaptation_type == "widen":
                # Widen grid spacing for high volatility
                adjustment_factor = min(volatility_ratio, self.volatility_grid_adjustment_factor)
                new_top = self.central_price * (1 + (current_top / self.central_price - 1) * adjustment_factor)
                new_bottom = self.central_price * (1 - (1 - current_bottom / self.central_price) * adjustment_factor)
            else:  # narrow
                # Narrow grid spacing for low volatility
                adjustment_factor = max(1 / volatility_ratio, 1 / self.volatility_grid_adjustment_factor)
                new_top = self.central_price * (1 + (current_top / self.central_price - 1) / adjustment_factor)
                new_bottom = self.central_price * (1 - (1 - current_bottom / self.central_price) / adjustment_factor)
            
            # Recalculate price grids with new range
            self.price_grids, _ = self._calculate_price_grids_with_range(new_bottom, new_top)
            
            # Update sorted buy/sell grids
            if self.strategy_type == StrategyType.SIMPLE_GRID:
                self.sorted_buy_grids = [price_grid for price_grid in self.price_grids if price_grid <= self.central_price]
                self.sorted_sell_grids = [price_grid for price_grid in self.price_grids if price_grid > self.central_price]
            elif self.strategy_type == StrategyType.HEDGED_GRID:
                self.sorted_buy_grids = self.price_grids[:-1]  # All except the top grid
                self.sorted_sell_grids = self.price_grids[1:]  # All except the bottom grid
            
            # Create new grid levels with states transferred from old levels where possible
            old_grid_levels = self.grid_levels.copy()
            new_grid_levels = {}
            
            for price in self.price_grids:
                # Find closest old grid level
                closest_old_price = min(old_grid_levels.keys(), key=lambda x: abs(x - price), default=None)
                
                if closest_old_price is not None and abs(closest_old_price - price) / price < 0.02:  # Within 2%
                    # Transfer state from closest old grid level
                    old_level = old_grid_levels[closest_old_price]
                    new_grid_levels[price] = GridLevel(price, old_level.state)
                    new_grid_levels[price].orders = old_level.orders
                    new_grid_levels[price].paired_buy_level = None  # Will be updated later
                    new_grid_levels[price].paired_sell_level = None  # Will be updated later
                else:
                    # Create new grid level with default state
                    if self.strategy_type == StrategyType.SIMPLE_GRID:
                        state = GridCycleState.READY_TO_BUY if price <= self.central_price else GridCycleState.READY_TO_SELL
                    else:  # HEDGED_GRID
                        state = GridCycleState.READY_TO_BUY_OR_SELL if price != self.price_grids[-1] else GridCycleState.READY_TO_SELL
                    new_grid_levels[price] = GridLevel(price, state)
            
            # Update grid levels
            self.grid_levels = new_grid_levels
            
            # Update last adjustment time
            self.last_grid_adjustment_time = time.time()
            
            # Log the adaptation
            self.logger.info(f"Grid spacing adapted: {adaptation_type} due to volatility change")
            self.logger.info(f"Current volatility: {current_volatility:.4f}%, Average volatility: {avg_volatility:.4f}%")
            self.logger.info(f"New price grids: {self.price_grids}")
            
            # Publish grid adaptation event
            await self._publish_grid_adaptation_event(current_volatility, avg_volatility, adaptation_type)
            
        finally:
            self.is_grid_adjustment_active = False
    
    def _calculate_price_grids_with_range(self, bottom_range: float, top_range: float) -> Tuple[List[float], float]:
        """
        Calculate price grids with a specific range.
        
        Args:
            bottom_range: The bottom price of the grid range
            top_range: The top price of the grid range
            
        Returns:
            Tuple containing the list of grid prices and the central price
        """
        num_grids = self.config_manager.get_num_grids()
        spacing_type = self.config_manager.get_spacing_type()
        
        if spacing_type == SpacingType.ARITHMETIC:
            grids = np.linspace(bottom_range, top_range, num_grids)
            central_price = (top_range + bottom_range) / 2
        
        elif spacing_type == SpacingType.GEOMETRIC:
            grids = []
            ratio = (top_range / bottom_range) ** (1 / (num_grids - 1))
            current_price = bottom_range
            
            for _ in range(num_grids):
                grids.append(current_price)
                current_price *= ratio
                
            central_index = len(grids) // 2
            if num_grids % 2 == 0:
                central_price = (grids[central_index - 1] + grids[central_index]) / 2
            else:
                central_price = grids[central_index]
        
        else:
            raise ValueError(f"Unsupported spacing type: {spacing_type}")
        
        return grids, central_price
    
    def _calculate_price_grids_and_central_price(self, center_price: Optional[float] = None) -> Tuple[List[float], float]:
        """
        Calculate price grids and central price, optionally centered around a specific price.
        
        Args:
            center_price: Optional center price to build grids around
            
        Returns:
            Tuple containing the list of grid prices and the central price
        """
        bottom_range, top_range, num_grids, spacing_type = self._extract_grid_config()
        
        if center_price is not None:
            # Adjust range to be centered around the provided price
            range_width = top_range - bottom_range
            bottom_range = center_price - range_width / 2
            top_range = center_price + range_width / 2
        
        if spacing_type == SpacingType.ARITHMETIC:
            grids = np.linspace(bottom_range, top_range, num_grids)
            central_price = center_price if center_price is not None else (top_range + bottom_range) / 2
        
        elif spacing_type == SpacingType.GEOMETRIC:
            grids = []
            ratio = (top_range / bottom_range) ** (1 / (num_grids - 1))
            current_price = bottom_range
            
            for _ in range(num_grids):
                grids.append(current_price)
                current_price *= ratio
                
            if center_price is not None:
                central_price = center_price
            else:
                central_index = len(grids) // 2
                if num_grids % 2 == 0:
                    central_price = (grids[central_index - 1] + grids[central_index]) / 2
                else:
                    central_price = grids[central_index]
        
        else:
            raise ValueError(f"Unsupported spacing type: {spacing_type}")
        
        return grids, central_price
    
    async def get_order_size_for_grid_level_with_leverage(
        self,
        total_balance: float,
        current_price: float
    ) -> float:
        """
        Calculates the order size for a grid level based on the total balance, leverage, and current price.
        
        Args:
            total_balance: The total balance available for trading
            current_price: The current price of the trading pair
            
        Returns:
            The calculated order size as a float
        """
        # Get leverage from config
        leverage = self.config_manager.get_leverage()
        
        # Calculate base order size using parent method
        base_order_size = self.get_order_size_for_grid_level(total_balance, current_price)
        
        # Apply leverage factor
        leveraged_order_size = base_order_size * leverage
        
        # Check if small capital optimization is enabled
        if self.small_capital_optimization_enabled:
            # Ensure minimum order value
            order_value = leveraged_order_size * current_price
            if order_value < self.min_order_value:
                leveraged_order_size = self.min_order_value / current_price
        
        # Check if the order size is within risk limits
        is_within_limits, risk_info = await self.risk_manager.is_position_within_risk_limits(
            "long",  # Default to long for size calculation
            leveraged_order_size,
            current_price
        )
        
        if not is_within_limits:
            # Adjust order size to maximum allowed
            max_size = risk_info.get("max_size", leveraged_order_size)
            self.logger.warning(f"Order size adjusted from {leveraged_order_size} to {max_size} due to risk limits")
            return max_size
        
        return leveraged_order_size
    
    async def _publish_grid_adjustment_event(self, old_center: float, new_center: float, adjustment_type: str) -> None:
        """
        Publish grid adjustment event to the event bus.
        
        Args:
            old_center: Previous central price
            new_center: New central price
            adjustment_type: Type of adjustment
        """
        if self.event_bus:
            await self.event_bus.publish(
                "grid_adjustment",
                {
                    "old_center": old_center,
                    "new_center": new_center,
                    "adjustment_type": adjustment_type,
                    "price_grids": self.price_grids,
                    "timestamp": time.time()
                }
            )
    
    async def _publish_grid_adaptation_event(self, current_volatility: float, avg_volatility: float, adaptation_type: str) -> None:
        """
        Publish grid adaptation event to the event bus.
        
        Args:
            current_volatility: Current market volatility
            avg_volatility: Average market volatility
            adaptation_type: Type of adaptation
        """
        if self.event_bus:
            await self.event_bus.publish(
                "grid_adaptation",
                {
                    "current_volatility": current_volatility,
                    "avg_volatility": avg_volatility,
                    "adaptation_type": adaptation_type,
                    "price_grids": self.price_grids,
                    "timestamp": time.time()
                }
            )
    
    async def shutdown(self) -> None:
        """
        Shutdown the dynamic grid manager, cancelling all background tasks.
        """
        self.logger.info("Shutting down DynamicGridManager...")
        
        # Cancel all grid adjustment tasks
        for task in self.grid_adjustment_tasks:
            task.cancel()
        
        # Wait for all tasks to complete
        if self.grid_adjustment_tasks:
            await asyncio.gather(*self.grid_adjustment_tasks, return_exceptions=True)