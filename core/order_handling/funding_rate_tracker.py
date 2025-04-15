import logging
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
import numpy as np
from decimal import Decimal

from config.config_manager import ConfigManager
from core.services.exchange_interface import ExchangeInterface
from core.bot_management.event_bus import EventBus

class FundingEvents:
    """
    Defines funding-related event types for the EventBus.
    """
    FUNDING_RATE_UPDATE = "funding_rate_update"
    FUNDING_PAYMENT_RECEIVED = "funding_payment_received"
    FUNDING_PAYMENT_PAID = "funding_payment_paid"
    UPCOMING_FUNDING_NOTIFICATION = "upcoming_funding_notification"
    FUNDING_TREND_CHANGE = "funding_trend_change"

class FundingRateTracker:
    """
    Tracks and analyzes funding rates for perpetual futures contracts.
    
    Responsibilities:
    1. Monitor and analyze funding rates
    2. Calculate funding payments
    3. Provide funding rate forecasting
    4. Adjust strategy based on funding trends
    5. Notify about upcoming funding events
    6. Integration with the ExchangeService
    7. Event publishing for funding rate events
    """
    
    def __init__(
        self,
        config_manager: ConfigManager,
        exchange_service: ExchangeInterface,
        event_bus: EventBus
    ):
        """
        Initialize the FundingRateTracker.
        
        Args:
            config_manager: Configuration manager instance
            exchange_service: Exchange service instance
            event_bus: Event bus for publishing funding events
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config_manager = config_manager
        self.exchange_service = exchange_service
        self.event_bus = event_bus
        
        # Validate that we're in futures mode with perpetual contracts
        if not self.config_manager.is_futures_market():
            raise ValueError("FundingRateTracker can only be used with futures markets")
        if self.config_manager.get_contract_type() != "perpetual":
            raise ValueError("FundingRateTracker can only be used with perpetual contracts")
        
        # Get trading pair
        self.base_currency = self.config_manager.get_base_currency()
        self.quote_currency = self.config_manager.get_quote_currency()
        self.pair = f"{self.base_currency}/{self.quote_currency}"
        self.exchange_pair = self.pair.replace("/", "")
        
        # Initialize funding rate tracking
        self.current_funding_rate = 0.0
        self.next_funding_time = None
        self.funding_interval = 8 * 60 * 60  # 8 hours in seconds (Bybit standard)
        self.funding_history = deque(maxlen=72)  # Store 72 funding rates (24 hours for 8-hour intervals)
        self.funding_payments = []
        
        # Load configuration settings
        self._load_config_settings()
        
        # Background tasks
        self.tracking_tasks = set()
        
        self.logger.info(f"Initialized FundingRateTracker for {self.pair}")
    
    def _load_config_settings(self) -> None:
        """
        Load configuration settings from the ConfigManager.
        """
        # Monitoring settings
        self.monitoring_enabled = self.config_manager.is_funding_rate_monitoring_enabled()
        self.update_interval = self.config_manager.get_funding_rate_update_interval()
        self.notification_threshold = self.config_manager.get_funding_rate_notification_threshold()
        
        # Threshold settings
        self.high_funding_threshold = self.config_manager.get_funding_rate_high_threshold()
        self.extreme_funding_threshold = self.config_manager.get_funding_rate_extreme_threshold()
        self.cumulative_threshold = self.config_manager.get_funding_rate_cumulative_threshold()
        
        # Strategy adjustment settings
        self.strategy_adjustment_enabled = self.config_manager.is_funding_rate_strategy_adjustment_enabled()
        self.reduce_exposure_threshold = self.config_manager.get_funding_rate_reduce_exposure_threshold()
        self.reverse_position_threshold = self.config_manager.get_funding_rate_reverse_position_threshold()
        self.max_adjustment_percentage = self.config_manager.get_funding_rate_max_adjustment_percentage()
        
        # Forecasting settings
        self.forecasting_enabled = self.config_manager.is_funding_rate_forecasting_enabled()
        self.forecast_window = self.config_manager.get_funding_rate_forecast_window()
        self.min_history_periods = self.config_manager.get_funding_rate_min_history_periods()
        self.confidence_threshold = self.config_manager.get_funding_rate_confidence_threshold()
        
        # Auto-hedge settings
        self.auto_hedge_enabled = self.config_manager.is_funding_rate_auto_hedge_enabled()
        self.hedge_threshold = self.config_manager.get_funding_rate_hedge_threshold()
        self.max_hedge_ratio = self.config_manager.get_funding_rate_max_hedge_ratio()
        self.min_funding_duration = self.config_manager.get_funding_rate_min_funding_duration()
        
        # Trend detection settings
        self.trend_detection_window = self.min_history_periods
        self.significant_trend_threshold = self.high_funding_threshold / 5  # 20% of high threshold
        
        # Derived settings
        self.low_funding_threshold = -self.high_funding_threshold
        
        self.logger.info(f"Loaded funding rate settings: update interval={self.update_interval}s, "
                        f"notification threshold={self.notification_threshold}s, "
                        f"high threshold={self.high_funding_threshold:.6f}")
    
    async def initialize(self) -> None:
        """
        Initialize the funding rate tracker by fetching current funding rate and starting tracking.
        """
        try:
            # Fetch current funding rate
            await self._update_funding_rate()
            
            # Start funding rate tracking
            self._start_funding_rate_tracking()
            
            self.logger.info(f"FundingRateTracker initialized successfully for {self.pair}")
        except Exception as e:
            self.logger.error(f"Error initializing FundingRateTracker: {e}")
            raise
    
    async def _update_funding_rate(self) -> None:
        """
        Fetch and update the current funding rate and next funding time.
        """
        try:
            funding_info = await self.exchange_service.get_funding_rate(self.exchange_pair)
            
            # Extract funding rate
            self.current_funding_rate = float(funding_info.get('fundingRate', 0.0))
            
            # Extract next funding time
            next_funding_timestamp = funding_info.get('nextFundingTime', 0)
            if next_funding_timestamp > 0:
                self.next_funding_time = datetime.fromtimestamp(next_funding_timestamp / 1000)  # Convert from milliseconds
            else:
                # If not provided, estimate based on Bybit's 8-hour schedule (00:00, 08:00, 16:00 UTC)
                now = datetime.utcnow()
                hours = now.hour
                next_funding_hour = (hours // 8 + 1) * 8 % 24
                next_funding_time = now.replace(hour=next_funding_hour, minute=0, second=0, microsecond=0)
                if next_funding_time <= now:
                    next_funding_time += timedelta(days=1)
                self.next_funding_time = next_funding_time
            
            # Add to history
            self.funding_history.append({
                'timestamp': datetime.utcnow(),
                'rate': self.current_funding_rate,
                'next_funding_time': self.next_funding_time
            })
            
            self.logger.info(f"Updated funding rate for {self.pair}: {self.current_funding_rate:.6f}, next funding at {self.next_funding_time}")
            
            # Publish funding rate update event
            await self.event_bus.publish(
                FundingEvents.FUNDING_RATE_UPDATE,
                {
                    'pair': self.pair,
                    'funding_rate': self.current_funding_rate,
                    'next_funding_time': self.next_funding_time.isoformat() if self.next_funding_time else None,
                    'timestamp': datetime.utcnow().isoformat()
                }
            )
            
            # Check if we need to notify about upcoming funding
            await self._check_upcoming_funding()
            
            # Analyze funding rate trends
            await self._analyze_funding_trends()
            
        except Exception as e:
            self.logger.error(f"Error updating funding rate: {e}")
    
    def _start_funding_rate_tracking(self) -> None:
        """
        Start background task for tracking funding rate updates.
        """
        tracking_task = asyncio.create_task(self._track_funding_rates())
        self.tracking_tasks.add(tracking_task)
        tracking_task.add_done_callback(self.tracking_tasks.discard)
    
    async def _track_funding_rates(self) -> None:
        """
        Periodically update funding rate information.
        """
        try:
            while True:
                await self._update_funding_rate()
                
                # Calculate time until next check
                # Check more frequently as we approach funding time
                now = datetime.utcnow()
                if self.next_funding_time:
                    time_to_funding = (self.next_funding_time - now).total_seconds()
                    if time_to_funding < 300:  # Less than 5 minutes to funding
                        wait_time = 60  # Check every minute
                    elif time_to_funding < 1800:  # Less than 30 minutes to funding
                        wait_time = 300  # Check every 5 minutes
                    else:
                        wait_time = self.update_interval  # Use configured interval
                else:
                    wait_time = self.update_interval  # Use configured interval
                
                await asyncio.sleep(wait_time)
        except asyncio.CancelledError:
            self.logger.info("Funding rate tracking task cancelled")
        except Exception as e:
            self.logger.error(f"Error in funding rate tracking: {e}")
    
    async def _check_upcoming_funding(self) -> None:
        """
        Check if funding is approaching and notify if needed.
        """
        if not self.next_funding_time:
            return
            
        now = datetime.utcnow()
        time_to_funding = (self.next_funding_time - now).total_seconds()
        
        # Notify if within threshold and we haven't notified recently
        if (time_to_funding <= self.notification_threshold and 
            time.time() - self.last_notification_time > self.notification_threshold):
            
            # Get positions to determine if we'll pay or receive funding
            positions = await self.exchange_service.get_positions(self.exchange_pair)
            
            # Calculate estimated funding payment
            estimated_payment = await self._calculate_estimated_funding_payment(positions)
            
            # Publish notification event
            await self.event_bus.publish(
                FundingEvents.UPCOMING_FUNDING_NOTIFICATION,
                {
                    'pair': self.pair,
                    'funding_rate': self.current_funding_rate,
                    'funding_time': self.next_funding_time.isoformat(),
                    'time_to_funding_minutes': int(time_to_funding / 60),
                    'estimated_payment': estimated_payment,
                    'will_pay': estimated_payment < 0,
                    'timestamp': now.isoformat()
                }
            )
            
            self.last_notification_time = time.time()
            self.logger.info(f"Sent upcoming funding notification for {self.pair}, funding in {int(time_to_funding / 60)} minutes")
    
    async def _calculate_estimated_funding_payment(self, positions: List[Dict[str, Any]]) -> float:
        """
        Calculate the estimated funding payment based on current positions and funding rate.
        
        Args:
            positions: List of position data from the exchange
            
        Returns:
            Estimated funding payment (positive for receiving, negative for paying)
        """
        total_payment = 0.0
        
        for position in positions:
            # Skip positions with zero size
            size = float(position.get('size', 0))
            if size == 0:
                continue
            
            # Get position details
            position_side = position.get('side', '').lower()
            entry_price = float(position.get('entryPrice', 0))
            
            # Calculate position value
            position_value = size * entry_price
            
            # Calculate funding payment
            # For long positions: negative funding rate means receiving payment
            # For short positions: positive funding rate means receiving payment
            if position_side == 'long':
                payment = -position_value * self.current_funding_rate
            else:  # short
                payment = position_value * self.current_funding_rate
            
            total_payment += payment
        
        return total_payment
    
    async def _analyze_funding_trends(self) -> None:
        """
        Analyze funding rate trends and detect significant changes.
        """
        if len(self.funding_history) < self.trend_detection_window:
            return
        
        # Get recent funding rates
        recent_rates = [entry['rate'] for entry in list(self.funding_history)[-self.trend_detection_window:]]
        
        # Calculate moving averages for trend detection
        if len(recent_rates) >= 3:
            short_ma = np.mean(recent_rates[-3:])
            long_ma = np.mean(recent_rates)
            
            # Detect trend change
            trend_change = abs(short_ma - long_ma) > self.significant_trend_threshold
            
            if trend_change:
                trend_direction = "increasing" if short_ma > long_ma else "decreasing"
                
                await self.event_bus.publish(
                    FundingEvents.FUNDING_TREND_CHANGE,
                    {
                        'pair': self.pair,
                        'trend_direction': trend_direction,
                        'short_term_average': short_ma,
                        'long_term_average': long_ma,
                        'recent_rates': recent_rates,
                        'timestamp': datetime.utcnow().isoformat()
                    }
                )
                
                self.logger.info(f"Detected funding rate trend change for {self.pair}: {trend_direction}")
                
                # If strategy adjustment is enabled, suggest adjustments
                if self.strategy_adjustment_enabled:
                    await self._suggest_strategy_adjustments(trend_direction, short_ma)
    
    async def _suggest_strategy_adjustments(self, trend_direction: str, current_average: float) -> None:
        """
        Suggest strategy adjustments based on funding rate trends.
        
        Args:
            trend_direction: Direction of the trend ('increasing' or 'decreasing')
            current_average: Current average funding rate
        """
        # High positive funding rates favor short positions
        # High negative funding rates favor long positions
        
        adjustment = None
        adjustment_percentage = 0.0
        
        if current_average > self.extreme_funding_threshold:
            # Extreme positive funding - shorts pay longs
            adjustment = "Consider significant reduction in long exposure or increasing short exposure"
            adjustment_percentage = self.max_adjustment_percentage
        elif current_average > self.reduce_exposure_threshold:
            # High positive funding - shorts pay longs
            adjustment = "Consider reducing long exposure or increasing short exposure"
            adjustment_percentage = self.max_adjustment_percentage * 0.5
        elif current_average < -self.extreme_funding_threshold:
            # Extreme negative funding - longs pay shorts
            adjustment = "Consider significant increase in long exposure or reducing short exposure"
            adjustment_percentage = self.max_adjustment_percentage
        elif current_average < -self.reduce_exposure_threshold:
            # High negative funding - longs pay shorts
            adjustment = "Consider increasing long exposure or reducing short exposure"
            adjustment_percentage = self.max_adjustment_percentage * 0.5
        
        if adjustment:
            self.logger.info(f"Strategy adjustment suggestion for {self.pair}: {adjustment} "
                           f"(suggested adjustment: {adjustment_percentage:.1%})")
            
            # If auto-hedge is enabled and funding rate is extreme, consider hedging
            if self.auto_hedge_enabled and abs(current_average) > self.hedge_threshold:
                hedge_direction = "short" if current_average > 0 else "long"
                hedge_ratio = min(abs(current_average) / self.extreme_funding_threshold * self.max_hedge_ratio,
                                 self.max_hedge_ratio)
                
                self.logger.info(f"Auto-hedge suggestion: Consider {hedge_ratio:.1%} hedge in {hedge_direction} direction")
    
    async def forecast_funding_rates(self) -> List[Dict[str, Any]]:
        """
        Forecast future funding rates based on historical data.
        
        Returns:
            List of forecasted funding rates with timestamps
        """
        if not self.forecasting_enabled or len(self.funding_history) < self.min_history_periods:
            return []
        
        # Get recent funding rates
        recent_rates = [entry['rate'] for entry in list(self.funding_history)[-self.min_history_periods:]]
        
        # Simple forecasting using linear regression
        x = np.arange(len(recent_rates)).reshape(-1, 1)
        y = np.array(recent_rates)
        
        # Fit linear regression model
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(x, y)
        
        # Forecast future rates
        future_x = np.arange(len(recent_rates), len(recent_rates) + self.forecast_window).reshape(-1, 1)
        forecasted_rates = model.predict(future_x)
        
        # Create forecast results
        forecast_results = []
        if self.next_funding_time:
            current_time = self.next_funding_time
            
            for i, rate in enumerate(forecasted_rates):
                forecast_time = current_time + timedelta(hours=8 * (i + 1))
                forecast_results.append({
                    'timestamp': forecast_time.isoformat(),
                    'forecasted_rate': float(rate),
                    'confidence': 1.0 - (i * 0.2)  # Decreasing confidence for further forecasts
                })
        
        return forecast_results
    
    async def get_funding_rate_history(self) -> List[Dict[str, Any]]:
        """
        Get the historical funding rates.
        
        Returns:
            List of historical funding rates
        """
        return list(self.funding_history)
    
    async def get_current_funding_info(self) -> Dict[str, Any]:
        """
        Get the current funding rate information.
        
        Returns:
            Dictionary with current funding information
        """
        return {
            'pair': self.pair,
            'current_rate': self.current_funding_rate,
            'next_funding_time': self.next_funding_time.isoformat() if self.next_funding_time else None,
            'time_to_next_funding': (self.next_funding_time - datetime.utcnow()).total_seconds() if self.next_funding_time else None,
            'funding_interval_hours': self.funding_interval / 3600,
            'last_updated': datetime.utcnow().isoformat()
        }
    
    async def record_funding_payment(self, payment_amount: float, timestamp: Optional[datetime] = None) -> None:
        """
        Record a funding payment that has occurred.
        
        Args:
            payment_amount: Amount of the funding payment (positive for received, negative for paid)
            timestamp: Timestamp of the payment, defaults to current time
        """
        if timestamp is None:
            timestamp = datetime.utcnow()
            
        payment_record = {
            'timestamp': timestamp,
            'amount': payment_amount,
            'funding_rate': self.current_funding_rate,
            'pair': self.pair
        }
        
        self.funding_payments.append(payment_record)
        
        # Publish appropriate event
        event_type = FundingEvents.FUNDING_PAYMENT_RECEIVED if payment_amount > 0 else FundingEvents.FUNDING_PAYMENT_PAID
        
        await self.event_bus.publish(
            event_type,
            {
                'pair': self.pair,
                'amount': payment_amount,
                'funding_rate': self.current_funding_rate,
                'timestamp': timestamp.isoformat(),
                'payment_record': payment_record
            }
        )
        
        self.logger.info(f"Recorded funding payment for {self.pair}: {payment_amount:.6f}")
    
    async def get_funding_payment_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of funding payments.
        
        Returns:
            List of funding payment records
        """
        return self.funding_payments
    
    async def get_funding_payment_summary(self) -> Dict[str, Any]:
        """
        Get a summary of funding payments.
        
        Returns:
            Dictionary with funding payment summary
        """
        if not self.funding_payments:
            return {
                'pair': self.pair,
                'total_payments': 0,
                'total_received': 0,
                'total_paid': 0,
                'net_amount': 0,
                'count': 0
            }
            
        total_received = sum(payment['amount'] for payment in self.funding_payments if payment['amount'] > 0)
        total_paid = sum(payment['amount'] for payment in self.funding_payments if payment['amount'] < 0)
        
        return {
            'pair': self.pair,
            'total_payments': len(self.funding_payments),
            'total_received': total_received,
            'total_paid': abs(total_paid),
            'net_amount': total_received + total_paid,  # total_paid is negative
            'count': len(self.funding_payments)
        }
    
    async def shutdown(self) -> None:
        """
        Shutdown the funding rate tracker, cancelling all background tasks.
        """
        self.logger.info("Shutting down FundingRateTracker...")
        
        # Cancel all tracking tasks
        for task in self.tracking_tasks:
            task.cancel()
        
        # Wait for all tasks to complete
        if self.tracking_tasks:
            await asyncio.gather(*self.tracking_tasks, return_exceptions=True)
        
        self.logger.info("FundingRateTracker shutdown complete")