import pytest
import asyncio
from unittest.mock import patch, MagicMock
from config.config_manager import ConfigManager
from config.market_type import MarketType
from core.services.backtest_exchange_service import BacktestExchangeService

class TestBacktestExchangeServiceFutures:
    
    @pytest.fixture
    def mock_config_manager(self):
        config_manager = MagicMock(spec=ConfigManager)
        config_manager.get_market_type.return_value = MarketType.FUTURES
        config_manager.get_exchange_name.return_value = "bybit"
        config_manager.get_margin_type.return_value = "isolated"
        config_manager.get_leverage.return_value = 3
        config_manager.is_hedge_mode_enabled.return_value = False
        config_manager.is_futures_market.return_value = True
        config_manager.get_contract_size.return_value = 1
        config_manager.get_historical_data_file.return_value = None
        return config_manager
    
    @pytest.fixture
    def exchange_service(self, mock_config_manager):
        with patch('ccxt.bybit', return_value=MagicMock()):
            service = BacktestExchangeService(mock_config_manager)
            # Mock the exchange methods needed for testing
            service.exchange.timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
            service.exchange.load_markets = MagicMock(return_value={'BTC/USDT:USDT': {}})
            return service
    
    def test_initialize_futures_settings(self, exchange_service):
        # Assert that futures settings were initialized
        assert exchange_service.leverage == 3
        assert exchange_service.margin_type == "isolated"
        assert exchange_service.hedge_mode is False
        assert exchange_service.contract_size == 1
        assert exchange_service.positions == {}
    
    @pytest.mark.asyncio
    async def test_set_leverage(self, exchange_service):
        # Arrange
        pair = "BTC/USDT:USDT"
        leverage = 5
        margin_mode = "cross"
        
        # Act
        result = await exchange_service.set_leverage(pair, leverage, margin_mode)
        
        # Assert
        assert exchange_service.leverage == 5
        assert exchange_service.margin_type == "cross"
        assert pair in exchange_service.positions
        assert exchange_service.positions[pair]['leverage'] == 5
        assert exchange_service.positions[pair]['margin_mode'] == "cross"
        assert result['leverage'] == 5
        assert result['marginMode'] == "cross"
        assert result['pair'] == pair
        assert result['success'] is True
    
    @pytest.mark.asyncio
    async def test_get_positions_empty(self, exchange_service):
        # Act
        positions = await exchange_service.get_positions()
        
        # Assert
        assert positions == []
    
    @pytest.mark.asyncio
    async def test_get_positions_with_data(self, exchange_service):
        # Arrange
        pair = "BTC/USDT:USDT"
        exchange_service.positions[pair] = {
            'leverage': 3,
            'margin_mode': 'isolated',
            'long': {
                'symbol': pair,
                'side': 'long',
                'size': 0.1,
                'entryPrice': 50000,
                'markPrice': 51000,
                'leverage': 3
            },
            'short': None
        }
        
        # Act
        positions = await exchange_service.get_positions(pair)
        
        # Assert
        assert len(positions) == 1
        assert positions[0]['symbol'] == pair
        assert positions[0]['side'] == 'long'
        assert positions[0]['size'] == 0.1
    
    @pytest.mark.asyncio
    async def test_close_position(self, exchange_service):
        # Arrange
        pair = "BTC/USDT:USDT"
        exchange_service.positions[pair] = {
            'leverage': 3,
            'margin_mode': 'isolated',
            'long': {
                'symbol': pair,
                'side': 'long',
                'size': 0.1,
                'entryPrice': 50000,
                'markPrice': 51000,
                'leverage': 3
            },
            'short': None
        }
        
        # Act
        result = await exchange_service.close_position(pair)
        
        # Assert
        assert result['status'] == 'closed'
        assert len(result['positions']) == 1
        assert result['positions'][0]['side'] == 'long'
        assert exchange_service.positions[pair]['long'] is None
    
    @pytest.mark.asyncio
    async def test_close_position_specific_side(self, exchange_service):
        # Arrange
        pair = "BTC/USDT:USDT"
        exchange_service.positions[pair] = {
            'leverage': 3,
            'margin_mode': 'isolated',
            'long': {
                'symbol': pair,
                'side': 'long',
                'size': 0.1,
                'entryPrice': 50000,
                'markPrice': 51000,
                'leverage': 3
            },
            'short': {
                'symbol': pair,
                'side': 'short',
                'size': 0.2,
                'entryPrice': 52000,
                'markPrice': 51000,
                'leverage': 3
            }
        }
        
        # Act
        result = await exchange_service.close_position(pair, 'short')
        
        # Assert
        assert result['status'] == 'closed'
        assert len(result['positions']) == 1
        assert result['positions'][0]['side'] == 'short'
        assert exchange_service.positions[pair]['short'] is None
        assert exchange_service.positions[pair]['long'] is not None  # Long position should remain
    
    @pytest.mark.asyncio
    async def test_close_position_no_position(self, exchange_service):
        # Arrange
        pair = "BTC/USDT:USDT"
        exchange_service.positions[pair] = {
            'leverage': 3,
            'margin_mode': 'isolated',
            'long': None,
            'short': None
        }
        
        # Act
        result = await exchange_service.close_position(pair)
        
        # Assert
        assert result['status'] == 'no_position'
    
    @pytest.mark.asyncio
    async def test_get_funding_rate(self, exchange_service):
        # Arrange
        pair = "BTC/USDT:USDT"
        
        # Act
        result = await exchange_service.get_funding_rate(pair)
        
        # Assert
        assert result['pair'] == pair
        assert 'fundingRate' in result
        assert 'fundingTimestamp' in result
        assert 'nextFundingTimestamp' in result
    
    @pytest.mark.asyncio
    async def test_get_contract_specifications(self, exchange_service):
        # Arrange
        pair = "BTC/USDT:USDT"
        
        # Act
        result = await exchange_service.get_contract_specifications(pair)
        
        # Assert
        assert result['pair'] == pair
        assert result['contract_size'] == 1
        assert 'price_precision' in result
        assert 'amount_precision' in result
        assert 'minimum_amount' in result
        assert 'maximum_amount' in result
        assert 'maintenance_margin_rate' in result
        assert 'is_inverse' in result
        assert 'is_linear' in result
        assert result['settlement_currency'] == 'USDT'