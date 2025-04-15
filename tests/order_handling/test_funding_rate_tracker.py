import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, AsyncMock
import numpy as np

from config.config_manager import ConfigManager
from core.services.exchange_interface import ExchangeInterface
from core.bot_management.event_bus import EventBus
from core.order_handling.funding_rate_tracker import FundingRateTracker, FundingEvents

@pytest.fixture
def mock_config_manager():
    config_manager = MagicMock(spec=ConfigManager)
    config_manager.is_futures_market.return_value = True
    config_manager.get_contract_type.return_value = "perpetual"
    config_manager.get_base_currency.return_value = "BTC"
    config_manager.get_quote_currency.return_value = "USDT"
    return config_manager

@pytest.fixture
def mock_exchange_service():
    exchange_service = AsyncMock(spec=ExchangeInterface)
    
    # Mock funding rate response
    funding_rate_response = {
        'fundingRate': 0.0001,  # 0.01% funding rate
        'nextFundingTime': int((datetime.utcnow() + timedelta(hours=4)).timestamp() * 1000)  # 4 hours from now
    }
    exchange_service.get_funding_rate.return_value = funding_rate_response
    
    # Mock positions response
    positions_response = [
        {
            'symbol': 'BTCUSDT',
            'side': 'long',
            'size': 0.1,
            'entryPrice': 50000,
            'leverage': 5,
            'marginType': 'isolated',
            'unrealizedPnl': 100,
            'realizedPnl': 50
        }
    ]
    exchange_service.get_positions.return_value = positions_response
    
    return exchange_service

@pytest.fixture
def mock_event_bus():
    event_bus = AsyncMock(spec=EventBus)
    return event_bus

@pytest.fixture
async def funding_rate_tracker(mock_config_manager, mock_exchange_service, mock_event_bus):
    tracker = FundingRateTracker(
        config_manager=mock_config_manager,
        exchange_service=mock_exchange_service,
        event_bus=mock_event_bus
    )
    
    # Patch the _start_funding_rate_tracking method to prevent background tasks
    with patch.object(tracker, '_start_funding_rate_tracking'):
        await tracker.initialize()
        yield tracker

class TestFundingRateTracker:
    
    async def test_initialization(self, funding_rate_tracker, mock_exchange_service):
        """Test that the FundingRateTracker initializes correctly."""
        assert funding_rate_tracker.pair == "BTC/USDT"
        assert funding_rate_tracker.exchange_pair == "BTCUSDT"
        assert funding_rate_tracker.current_funding_rate == 0.0001
        assert funding_rate_tracker.next_funding_time is not None
        
        # Verify that the exchange service was called to get the funding rate
        mock_exchange_service.get_funding_rate.assert_called_once_with("BTCUSDT")
    
    async def test_update_funding_rate(self, funding_rate_tracker, mock_exchange_service, mock_event_bus):
        """Test that the funding rate is updated correctly."""
        # Update the mock to return a different funding rate
        new_funding_rate = 0.0002
        mock_exchange_service.get_funding_rate.return_value = {
            'fundingRate': new_funding_rate,
            'nextFundingTime': int((datetime.utcnow() + timedelta(hours=2)).timestamp() * 1000)
        }
        
        # Call the update method
        await funding_rate_tracker._update_funding_rate()
        
        # Verify the funding rate was updated
        assert funding_rate_tracker.current_funding_rate == new_funding_rate
        
        # Verify the event was published
        mock_event_bus.publish.assert_called_with(
            FundingEvents.FUNDING_RATE_UPDATE,
            {
                'pair': "BTC/USDT",
                'funding_rate': new_funding_rate,
                'next_funding_time': funding_rate_tracker.next_funding_time.isoformat(),
                'timestamp': pytest.approx(datetime.utcnow().isoformat(), abs=timedelta(seconds=5))
            }
        )
    
    async def test_calculate_estimated_funding_payment(self, funding_rate_tracker, mock_exchange_service):
        """Test that funding payments are calculated correctly."""
        # Set up test data
        positions = [
            {
                'symbol': 'BTCUSDT',
                'side': 'long',
                'size': 0.1,
                'entryPrice': 50000,
                'leverage': 5
            }
        ]
        
        # For a long position with positive funding rate, payment should be negative
        funding_rate_tracker.current_funding_rate = 0.0001  # 0.01%
        payment = await funding_rate_tracker._calculate_estimated_funding_payment(positions)
        
        # Expected payment: -0.01% * (0.1 BTC * $50,000) = -$0.5
        expected_payment = -0.5
        assert payment == pytest.approx(expected_payment, abs=0.001)
        
        # For a short position with positive funding rate, payment should be positive
        positions[0]['side'] = 'short'
        payment = await funding_rate_tracker._calculate_estimated_funding_payment(positions)
        
        # Expected payment: 0.01% * (0.1 BTC * $50,000) = $0.5
        expected_payment = 0.5
        assert payment == pytest.approx(expected_payment, abs=0.001)
        
        # For a long position with negative funding rate, payment should be positive
        positions[0]['side'] = 'long'
        funding_rate_tracker.current_funding_rate = -0.0001  # -0.01%
        payment = await funding_rate_tracker._calculate_estimated_funding_payment(positions)
        
        # Expected payment: -(-0.01%) * (0.1 BTC * $50,000) = $0.5
        expected_payment = 0.5
        assert payment == pytest.approx(expected_payment, abs=0.001)
    
    async def test_check_upcoming_funding(self, funding_rate_tracker, mock_event_bus, mock_exchange_service):
        """Test that upcoming funding notifications are sent correctly."""
        # Set up test data
        funding_rate_tracker.current_funding_rate = 0.0001
        funding_rate_tracker.next_funding_time = datetime.utcnow() + timedelta(minutes=20)
        funding_rate_tracker.notification_threshold = 30 * 60  # 30 minutes
        funding_rate_tracker.last_notification_time = 0  # Ensure notification will be sent
        
        # Call the check method
        await funding_rate_tracker._check_upcoming_funding()
        
        # Verify the notification was sent
        mock_event_bus.publish.assert_called_with(
            FundingEvents.UPCOMING_FUNDING_NOTIFICATION,
            {
                'pair': "BTC/USDT",
                'funding_rate': 0.0001,
                'funding_time': funding_rate_tracker.next_funding_time.isoformat(),
                'time_to_funding_minutes': pytest.approx(20, abs=1),
                'estimated_payment': pytest.approx(-0.5, abs=0.001),
                'will_pay': True,
                'timestamp': pytest.approx(datetime.utcnow().isoformat(), abs=timedelta(seconds=5))
            }
        )
    
    async def test_analyze_funding_trends(self, funding_rate_tracker, mock_event_bus):
        """Test that funding trends are analyzed correctly."""
        # Set up test data with an increasing trend
        funding_rate_tracker.trend_detection_window = 6
        funding_rate_tracker.significant_trend_threshold = 0.00005
        
        # Create history with increasing rates
        funding_rate_tracker.funding_history.clear()
        for i in range(6):
            funding_rate_tracker.funding_history.append({
                'timestamp': datetime.utcnow() - timedelta(hours=8*(5-i)),
                'rate': 0.0001 + (i * 0.00002),
                'next_funding_time': datetime.utcnow() + timedelta(hours=i)
            })
        
        # Call the analyze method
        await funding_rate_tracker._analyze_funding_trends()
        
        # Verify the trend change event was published
        mock_event_bus.publish.assert_called_with(
            FundingEvents.FUNDING_TREND_CHANGE,
            {
                'pair': "BTC/USDT",
                'trend_direction': "increasing",
                'short_term_average': pytest.approx(0.00016, abs=0.00001),
                'long_term_average': pytest.approx(0.00013, abs=0.00001),
                'recent_rates': [0.0001, 0.00012, 0.00014, 0.00016, 0.00018, 0.0002],
                'timestamp': pytest.approx(datetime.utcnow().isoformat(), abs=timedelta(seconds=5))
            }
        )
    
    async def test_record_funding_payment(self, funding_rate_tracker, mock_event_bus):
        """Test that funding payments are recorded correctly."""
        # Record a positive payment (received)
        payment_amount = 1.5
        await funding_rate_tracker.record_funding_payment(payment_amount)
        
        # Verify the payment was recorded
        assert len(funding_rate_tracker.funding_payments) == 1
        assert funding_rate_tracker.funding_payments[0]['amount'] == payment_amount
        
        # Verify the event was published
        mock_event_bus.publish.assert_called_with(
            FundingEvents.FUNDING_PAYMENT_RECEIVED,
            {
                'pair': "BTC/USDT",
                'amount': payment_amount,
                'funding_rate': funding_rate_tracker.current_funding_rate,
                'timestamp': pytest.approx(datetime.utcnow().isoformat(), abs=timedelta(seconds=5)),
                'payment_record': funding_rate_tracker.funding_payments[0]
            }
        )
        
        # Record a negative payment (paid)
        payment_amount = -2.0
        await funding_rate_tracker.record_funding_payment(payment_amount)
        
        # Verify the payment was recorded
        assert len(funding_rate_tracker.funding_payments) == 2
        assert funding_rate_tracker.funding_payments[1]['amount'] == payment_amount
        
        # Verify the event was published
        mock_event_bus.publish.assert_called_with(
            FundingEvents.FUNDING_PAYMENT_PAID,
            {
                'pair': "BTC/USDT",
                'amount': payment_amount,
                'funding_rate': funding_rate_tracker.current_funding_rate,
                'timestamp': pytest.approx(datetime.utcnow().isoformat(), abs=timedelta(seconds=5)),
                'payment_record': funding_rate_tracker.funding_payments[1]
            }
        )
    
    async def test_get_funding_payment_summary(self, funding_rate_tracker):
        """Test that funding payment summaries are calculated correctly."""
        # Add some test payments
        funding_rate_tracker.funding_payments = [
            {'timestamp': datetime.utcnow(), 'amount': 1.5, 'funding_rate': 0.0001, 'pair': "BTC/USDT"},
            {'timestamp': datetime.utcnow(), 'amount': -0.8, 'funding_rate': 0.0002, 'pair': "BTC/USDT"},
            {'timestamp': datetime.utcnow(), 'amount': 2.0, 'funding_rate': -0.0001, 'pair': "BTC/USDT"}
        ]
        
        # Get the summary
        summary = await funding_rate_tracker.get_funding_payment_summary()
        
        # Verify the summary
        assert summary['pair'] == "BTC/USDT"
        assert summary['total_payments'] == 3
        assert summary['total_received'] == 3.5
        assert summary['total_paid'] == 0.8
        assert summary['net_amount'] == 2.7
        assert summary['count'] == 3
    
    async def test_forecast_funding_rates(self, funding_rate_tracker):
        """Test that funding rate forecasting works correctly."""
        # Set up test data
        funding_rate_tracker.trend_detection_window = 6
        funding_rate_tracker.forecast_window = 3
        
        # Create history with increasing rates
        funding_rate_tracker.funding_history.clear()
        for i in range(6):
            funding_rate_tracker.funding_history.append({
                'timestamp': datetime.utcnow() - timedelta(hours=8*(5-i)),
                'rate': 0.0001 + (i * 0.00002),
                'next_funding_time': datetime.utcnow() + timedelta(hours=i)
            })
        
        # Set next funding time
        funding_rate_tracker.next_funding_time = datetime.utcnow() + timedelta(hours=4)
        
        # Mock sklearn for the test
        with patch('sklearn.linear_model.LinearRegression') as mock_lr:
            # Configure the mock
            mock_model = MagicMock()
            mock_model.predict.return_value = np.array([0.00022, 0.00024, 0.00026])
            mock_lr.return_value = mock_model
            
            # Get the forecast
            forecast = await funding_rate_tracker.forecast_funding_rates()
            
            # Verify the forecast
            assert len(forecast) == 3
            assert forecast[0]['forecasted_rate'] == 0.00022
            assert forecast[1]['forecasted_rate'] == 0.00024
            assert forecast[2]['forecasted_rate'] == 0.00026
            
            # Verify confidence decreases
            assert forecast[0]['confidence'] > forecast[1]['confidence']
            assert forecast[1]['confidence'] > forecast[2]['confidence']
    
    async def test_get_current_funding_info(self, funding_rate_tracker):
        """Test that current funding info is returned correctly."""
        # Set up test data
        funding_rate_tracker.current_funding_rate = 0.0001
        funding_rate_tracker.next_funding_time = datetime.utcnow() + timedelta(hours=4)
        
        # Get the info
        info = await funding_rate_tracker.get_current_funding_info()
        
        # Verify the info
        assert info['pair'] == "BTC/USDT"
        assert info['current_rate'] == 0.0001
        assert info['next_funding_time'] == funding_rate_tracker.next_funding_time.isoformat()
        assert info['time_to_next_funding'] == pytest.approx(4 * 3600, abs=5)
        assert info['funding_interval_hours'] == 8
    
    async def test_invalid_market_type(self, mock_config_manager, mock_exchange_service, mock_event_bus):
        """Test that the tracker raises an error for non-futures markets."""
        mock_config_manager.is_futures_market.return_value = False
        
        with pytest.raises(ValueError, match="FundingRateTracker can only be used with futures markets"):
            FundingRateTracker(
                config_manager=mock_config_manager,
                exchange_service=mock_exchange_service,
                event_bus=mock_event_bus
            )
    
    async def test_invalid_contract_type(self, mock_config_manager, mock_exchange_service, mock_event_bus):
        """Test that the tracker raises an error for non-perpetual contracts."""
        mock_config_manager.is_futures_market.return_value = True
        mock_config_manager.get_contract_type.return_value = "delivery"
        
        with pytest.raises(ValueError, match="FundingRateTracker can only be used with perpetual contracts"):
            FundingRateTracker(
                config_manager=mock_config_manager,
                exchange_service=mock_exchange_service,
                event_bus=mock_event_bus
            )