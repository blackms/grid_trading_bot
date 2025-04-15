import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal

from config.config_manager import ConfigManager
from core.services.exchange_interface import ExchangeInterface
from core.bot_management.event_bus import EventBus, Events

class PositionEvents:
    """
    Defines position-related event types for the EventBus.
    """
    POSITION_OPENED = "position_opened"
    POSITION_MODIFIED = "position_modified"
    POSITION_CLOSED = "position_closed"
    POSITION_LIQUIDATION_WARNING = "position_liquidation_warning"
    POSITION_PNL_UPDATE = "position_pnl_update"

class Position:
    """
    Represents a futures position with its properties and status.
    """
    def __init__(
        self,
        pair: str,
        position_side: str,
        size: float,
        entry_price: float,
        leverage: int,
        margin_type: str,
        initial_margin: float,
        liquidation_price: Optional[float] = None,
        unrealized_pnl: float = 0.0,
        realized_pnl: float = 0.0,
        position_id: Optional[str] = None
    ):
        self.pair = pair
        self.position_side = position_side  # 'long' or 'short'
        self.size = size
        self.entry_price = entry_price
        self.leverage = leverage
        self.margin_type = margin_type  # 'isolated' or 'cross'
        self.initial_margin = initial_margin
        self.liquidation_price = liquidation_price
        self.unrealized_pnl = unrealized_pnl
        self.realized_pnl = realized_pnl
        self.position_id = position_id
        self.is_open = True
        self.last_update_time = None
        self.funding_payments = []
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert position to dictionary for serialization.
        """
        return {
            "pair": self.pair,
            "position_side": self.position_side,
            "size": self.size,
            "entry_price": self.entry_price,
            "leverage": self.leverage,
            "margin_type": self.margin_type,
            "initial_margin": self.initial_margin,
            "liquidation_price": self.liquidation_price,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "position_id": self.position_id,
            "is_open": self.is_open,
            "last_update_time": self.last_update_time,
            "funding_payments": self.funding_payments
        }
    
    @classmethod
    def from_exchange_data(cls, exchange_position_data: Dict[str, Any]) -> 'Position':
        """
        Create a Position object from exchange API response data.
        
        Args:
            exchange_position_data: Position data from exchange API
            
        Returns:
            Position object
        """
        # Extract common fields, handling different exchange formats
        pair = exchange_position_data.get('symbol')
        
        # Normalize position side
        raw_side = exchange_position_data.get('side', '')
        position_side = raw_side.lower()
        
        # Extract size (could be called size, amount, or contracts)
        size = float(exchange_position_data.get('size', 
                    exchange_position_data.get('contracts',
                    exchange_position_data.get('positionAmt', 0))))
        
        # Handle negative size for short positions if not explicitly marked
        if size < 0 and position_side != 'short':
            position_side = 'short'
            size = abs(size)
        
        # Extract other fields
        entry_price = float(exchange_position_data.get('entryPrice', 0))
        leverage = int(exchange_position_data.get('leverage', 1))
        margin_type = exchange_position_data.get('marginType', 'isolated').lower()
        
        # Calculate or extract initial margin
        initial_margin = float(exchange_position_data.get('initialMargin', 
                              exchange_position_data.get('positionMargin', 0)))
        
        # If initial margin not provided, calculate it
        if initial_margin == 0 and entry_price > 0 and size > 0:
            initial_margin = (entry_price * size) / leverage
        
        # Extract liquidation price
        liquidation_price = float(exchange_position_data.get('liquidationPrice', 0))
        
        # Extract PnL information
        unrealized_pnl = float(exchange_position_data.get('unrealizedPnl', 0))
        realized_pnl = float(exchange_position_data.get('realizedPnl', 0))
        
        # Extract position ID if available
        position_id = exchange_position_data.get('id', 
                     exchange_position_data.get('positionId', None))
        
        return cls(
            pair=pair,
            position_side=position_side,
            size=size,
            entry_price=entry_price,
            leverage=leverage,
            margin_type=margin_type,
            initial_margin=initial_margin,
            liquidation_price=liquidation_price,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl,
            position_id=position_id
        )


class FuturesPositionManager:
    """
    Manages futures positions lifecycle, tracking, and coordination with other futures components.
    
    Responsibilities:
    1. Position opening with appropriate leverage
    2. Position modification as grid orders are filled
    3. Position tracking (unrealized and realized PnL)
    4. Position closing
    5. Integration with the ExchangeService
    6. Event publishing for position events
    """
    
    def __init__(
        self,
        config_manager: ConfigManager,
        exchange_service: ExchangeInterface,
        event_bus: EventBus
    ):
        """
        Initialize the FuturesPositionManager.
        
        Args:
            config_manager: Configuration manager instance
            exchange_service: Exchange service instance
            event_bus: Event bus for publishing position events
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config_manager = config_manager
        self.exchange_service = exchange_service
        self.event_bus = event_bus
        
        # Validate that we're in futures mode
        if not self.config_manager.is_futures_market():
            raise ValueError("FuturesPositionManager can only be used with futures markets")
        
        # Initialize position tracking
        self.positions: Dict[str, Position] = {}  # key: pair-side (e.g., "BTCUSDT-long")
        self.position_update_tasks = set()
        
        # Get trading pair
        self.base_currency = self.config_manager.get_base_currency()
        self.quote_currency = self.config_manager.get_quote_currency()
        self.pair = f"{self.base_currency}/{self.quote_currency}"
        
        # Get futures settings
        self.leverage = self.config_manager.get_leverage()
        self.margin_type = self.config_manager.get_margin_type()
        self.hedge_mode = self.config_manager.is_hedge_mode_enabled()
        
        # Risk management settings
        self.liquidation_protection_enabled = self.config_manager.is_liquidation_protection_enabled()
        self.liquidation_protection_threshold = self.config_manager.get_liquidation_protection_threshold()
        self.max_position_size = self.config_manager.get_max_position_size()
        
        self.logger.info(f"Initialized FuturesPositionManager for {self.pair} with leverage {self.leverage}x")
    
    async def initialize(self) -> None:
        """
        Initialize the position manager by setting leverage and loading existing positions.
        """
        try:
            # Set leverage on the exchange
            await self._set_leverage()
            
            # Load existing positions
            await self.load_positions()
            
            # Start position tracking
            self._start_position_tracking()
            
            self.logger.info(f"FuturesPositionManager initialized successfully for {self.pair}")
        except Exception as e:
            self.logger.error(f"Error initializing FuturesPositionManager: {e}")
            raise
    
    async def _set_leverage(self) -> None:
        """
        Set the leverage and margin type on the exchange.
        """
        try:
            exchange_pair = self.pair.replace("/", "")
            result = await self.exchange_service.set_leverage(
                pair=exchange_pair,
                leverage=self.leverage,
                margin_mode=self.margin_type
            )
            self.logger.info(f"Set leverage for {exchange_pair} to {self.leverage}x with {self.margin_type} margin: {result}")
        except Exception as e:
            self.logger.error(f"Failed to set leverage: {e}")
            raise
    
    async def load_positions(self) -> None:
        """
        Load existing positions from the exchange.
        """
        try:
            exchange_pair = self.pair.replace("/", "")
            positions_data = await self.exchange_service.get_positions(exchange_pair)
            
            for position_data in positions_data:
                # Skip positions with zero size
                size = float(position_data.get('size', 0))
                if size == 0:
                    continue
                
                position = Position.from_exchange_data(position_data)
                position_key = self._get_position_key(position.pair, position.position_side)
                self.positions[position_key] = position
                
                self.logger.info(f"Loaded existing position: {position_key} with size {position.size}")
            
            self.logger.info(f"Loaded {len(self.positions)} existing positions")
        except Exception as e:
            self.logger.error(f"Failed to load positions: {e}")
            raise
    
    def _get_position_key(self, pair: str, position_side: str) -> str:
        """
        Generate a unique key for a position based on pair and side.
        
        Args:
            pair: Trading pair
            position_side: Position side ('long' or 'short')
            
        Returns:
            Position key string
        """
        return f"{pair}-{position_side}"
    
    def _start_position_tracking(self) -> None:
        """
        Start background task for tracking position updates.
        """
        tracking_task = asyncio.create_task(self._track_positions())
        self.position_update_tasks.add(tracking_task)
        tracking_task.add_done_callback(self.position_update_tasks.discard)
    
    async def _track_positions(self) -> None:
        """
        Periodically update position information including PnL and check for liquidation risks.
        """
        try:
            while True:
                await self._update_positions()
                await asyncio.sleep(5)  # Update every 5 seconds
        except asyncio.CancelledError:
            self.logger.info("Position tracking task cancelled")
        except Exception as e:
            self.logger.error(f"Error in position tracking: {e}")
    
    async def _update_positions(self) -> None:
        """
        Update all tracked positions with latest data from the exchange.
        """
        if not self.positions:
            return
        
        try:
            exchange_pair = self.pair.replace("/", "")
            positions_data = await self.exchange_service.get_positions(exchange_pair)
            
            for position_data in positions_data:
                # Skip positions with zero size
                size = float(position_data.get('size', 0))
                if size == 0:
                    continue
                
                updated_position = Position.from_exchange_data(position_data)
                position_key = self._get_position_key(updated_position.pair, updated_position.position_side)
                
                # If we're tracking this position, update it
                if position_key in self.positions:
                    old_position = self.positions[position_key]
                    
                    # Check for significant changes in PnL
                    pnl_changed = abs(old_position.unrealized_pnl - updated_position.unrealized_pnl) > 0.01
                    
                    # Update the position
                    self.positions[position_key] = updated_position
                    
                    # Publish PnL update event if significant change
                    if pnl_changed:
                        await self.event_bus.publish(
                            PositionEvents.POSITION_PNL_UPDATE,
                            {
                                "pair": updated_position.pair,
                                "position_side": updated_position.position_side,
                                "unrealized_pnl": updated_position.unrealized_pnl,
                                "realized_pnl": updated_position.realized_pnl,
                                "entry_price": updated_position.entry_price,
                                "current_price": self._calculate_current_price(updated_position),
                                "position": updated_position.to_dict()
                            }
                        )
                    
                    # Check for liquidation risk
                    if self.liquidation_protection_enabled and updated_position.liquidation_price:
                        await self._check_liquidation_risk(updated_position)
                else:
                    # New position appeared that we weren't tracking
                    self.positions[position_key] = updated_position
                    self.logger.info(f"Started tracking new position: {position_key}")
                    
                    # Publish position opened event
                    await self.event_bus.publish(
                        PositionEvents.POSITION_OPENED,
                        {
                            "pair": updated_position.pair,
                            "position_side": updated_position.position_side,
                            "size": updated_position.size,
                            "entry_price": updated_position.entry_price,
                            "leverage": updated_position.leverage,
                            "position": updated_position.to_dict()
                        }
                    )
            
            # Check for positions that no longer exist on the exchange
            exchange_position_keys = {
                self._get_position_key(Position.from_exchange_data(p).pair, 
                                      Position.from_exchange_data(p).position_side): True 
                for p in positions_data if float(p.get('size', 0)) > 0
            }
            
            closed_positions = []
            for position_key, position in self.positions.items():
                if position_key not in exchange_position_keys:
                    closed_positions.append(position_key)
                    
                    # Publish position closed event
                    await self.event_bus.publish(
                        PositionEvents.POSITION_CLOSED,
                        {
                            "pair": position.pair,
                            "position_side": position.position_side,
                            "realized_pnl": position.realized_pnl,
                            "position": position.to_dict()
                        }
                    )
            
            # Remove closed positions from tracking
            for position_key in closed_positions:
                self.logger.info(f"Position closed: {position_key}")
                del self.positions[position_key]
                
        except Exception as e:
            self.logger.error(f"Error updating positions: {e}")
    
    def _calculate_current_price(self, position: Position) -> float:
        """
        Calculate the current price based on position data.
        This is an estimate since we don't have the actual mark price in the position data.
        
        Args:
            position: The position to calculate current price for
            
        Returns:
            Estimated current price
        """
        if position.size == 0 or position.entry_price == 0:
            return 0
            
        # For long positions: entry_price + (unrealized_pnl / size)
        # For short positions: entry_price - (unrealized_pnl / size)
        if position.position_side == 'long':
            return position.entry_price + (position.unrealized_pnl / position.size)
        else:
            return position.entry_price - (position.unrealized_pnl / position.size)
    
    async def _check_liquidation_risk(self, position: Position) -> None:
        """
        Check if a position is at risk of liquidation and publish warning event if needed.
        
        Args:
            position: The position to check
        """
        if not position.liquidation_price:
            return
            
        try:
            # Get current price
            exchange_pair = position.pair.replace("/", "")
            current_price = await self.exchange_service.get_current_price(exchange_pair)
            
            # Calculate distance to liquidation
            if position.position_side == 'long':
                distance_to_liquidation = (current_price - position.liquidation_price) / current_price
            else:
                distance_to_liquidation = (position.liquidation_price - current_price) / current_price
            
            # Check if we're below the threshold
            if distance_to_liquidation < self.liquidation_protection_threshold:
                await self.event_bus.publish(
                    PositionEvents.POSITION_LIQUIDATION_WARNING,
                    {
                        "pair": position.pair,
                        "position_side": position.position_side,
                        "current_price": current_price,
                        "liquidation_price": position.liquidation_price,
                        "distance_to_liquidation": distance_to_liquidation,
                        "threshold": self.liquidation_protection_threshold,
                        "position": position.to_dict()
                    }
                )
                self.logger.warning(
                    f"Liquidation warning for {position.pair} {position.position_side}: "
                    f"Current price: {current_price}, Liquidation price: {position.liquidation_price}, "
                    f"Distance: {distance_to_liquidation:.2%}"
                )
        except Exception as e:
            self.logger.error(f"Error checking liquidation risk: {e}")
    
    async def get_position(self, side: str) -> Optional[Dict[str, Any]]:
        """
        Get current position details for a specific side.
        
        Args:
            side: Position side ('buy'/'long' or 'sell'/'short')
            
        Returns:
            Position details or None if no position exists
        """
        # Normalize side
        normalized_side = side.lower()
        if normalized_side in ('buy', 'long'):
            position_side = 'long'
        elif normalized_side in ('sell', 'short'):
            position_side = 'short'
        else:
            raise ValueError(f"Invalid side: {side}. Must be one of: buy, long, sell, short")
        
        # Get position key
        position_key = self._get_position_key(self.pair.replace("/", ""), position_side)
        
        # Return position if it exists
        if position_key in self.positions:
            return self.positions[position_key].to_dict()
        
        return None
    
    async def get_all_positions(self) -> List[Dict[str, Any]]:
        """
        Get all current positions.
        
        Returns:
            List of position details
        """
        return [position.to_dict() for position in self.positions.values()]
    
    async def shutdown(self) -> None:
        """
        Shutdown the position manager, cancelling all background tasks.
        """
        self.logger.info("Shutting down FuturesPositionManager...")
        
        # Cancel all position update tasks
        for task in self.position_update_tasks:
            task.cancel()
        
        # Wait for all tasks to complete
        if self.position_update_tasks:
            await asyncio.gather(*self.position_update_tasks, return_exceptions=True)
            self.position_update_tasks.clear()
        
        self.logger.info("FuturesPositionManager shutdown complete")
