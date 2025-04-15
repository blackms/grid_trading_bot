import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from config.config_manager import ConfigManager
from config.market_type import MarketType
from core.services.live_exchange_service import LiveExchangeService
from core.services.exceptions import DataFetchError

class TestLiveExchangeServiceFutures:
    
    @pytest.fixture
    def mock_config_manager(self):
        config_manager = MagicMock(spec=ConfigManager)
        config_manager.get_market_type.return_value = MarketType.FUTURES
        config_manager.get_exchange_name.return_value = "bybit"
        config_manager.get_margin_type.return_value = "isolated"
        config_manager.get_leverage.return_value = 3
        config_manager.is_hedge_mode_enabled.return_value = False
        config_manager.is_futures_market.return_value = True
        return config_manager
    
    @pytest.fixture
    def mock_exchange(self):
        exchange = AsyncMock()
        exchange.set_margin_mode = AsyncMock()
        exchange.set_leverage = AsyncMock(return_value={"leverage": 3})
        exchange.set_position_mode = AsyncMock()
        exchange.fetch_positions = AsyncMock(return_value=[
            {
                "symbol": "BTC/USDT:USDT",
                "side": "long",
                "contracts": 0.1,
                "contractSize": 1,
                "entryPrice": 50000,
                "markPrice": 51000,
                "notional": 5000,
                "leverage": 3,
                "marginType": "isolated",
                "liquidationPrice": 40000,
                "unrealizedPnl": 100
            }
        ])
        exchange.create_order = AsyncMock(return_value={
            "id": "12345",
            "symbol": "BTC/USDT:USDT",
            "type": "market",
            "side": "sell",
            "amount": 0.1,
            "status": "closed"
        })
        exchange.fetch_funding_rate = AsyncMock(return_value={
            "symbol": "BTC/USDT:USDT",
            "markPrice": 51000,
            "indexPrice": 50900,
            "fundingRate": 0.0001,
            "fundingTimestamp": 1619712000000,
            "nextFundingTimestamp": 1619740800000
        })
        exchange.fetch_market = AsyncMock(return_value={
            "id": "BTCUSDT",
            "symbol": "BTC/USDT:USDT",
            "base": "BTC",
            "quote": "USDT",
            "settle": "USDT",
            "settleId": "USDT",
            "contractSize": 1,
            "precision": {"price": 0.5, "amount": 0.001},
            "limits": {
                "amount": {"min": 0.001, "max": 100},
                "cost": {"min": 5},
                "leverage": {"max": 100}
            },
            "info": {"maintMarginRate": 0.005},
            "inverse": False,
            "linear": True
        })
        return exchange
    
    @pytest.fixture
    def exchange_service(self, mock_config_manager, mock_exchange):
        with patch('os.getenv', return_value="dummy_key"), \
             patch('ccxtpro.bybit', return_value=mock_exchange):
            service = LiveExchangeService(mock_config_manager, is_paper_trading_activated=False)
            service.exchange = mock_exchange
            service._futures_initialized = False
            return service
    
    @pytest.mark.asyncio
    async def test_initialize_futures_settings(self, exchange_service, mock_exchange):
        # Act
        await exchange_service._ensure_futures_initialized("BTC/USDT:USDT")
        
        # Assert
        mock_exchange.set_margin_mode.assert_called_once_with("isolated", "BTC/USDT:USDT")
        mock_exchange.set_leverage.assert_called_once_with(3, "BTC/USDT:USDT")
        assert exchange_service._futures_initialized is True
    
    @pytest.mark.asyncio
    async def test_set_leverage(self, exchange_service, mock_exchange):
        # Arrange
        pair = "BTC/USDT:USDT"
        leverage = 5
        margin_mode = "cross"
        
        # Act
        result = await exchange_service.set_leverage(pair, leverage, margin_mode)
        
        # Assert
        mock_exchange.set_margin_mode.assert_called_once_with(margin_mode, pair)
        mock_exchange.set_leverage.assert_called_once_with(leverage, pair)
        assert result == {"leverage": 3}  # Mock return value
    
    @pytest.mark.asyncio
    async def test_get_positions(self, exchange_service, mock_exchange):
        # Act
        positions = await exchange_service.get_positions("BTC/USDT:USDT")
        
        # Assert
        mock_exchange.fetch_positions.assert_called_once_with("BTC/USDT:USDT")
        assert len(positions) == 1
        assert positions[0]["symbol"] == "BTC/USDT:USDT"
        assert positions[0]["side"] == "long"
        assert positions[0]["contracts"] == 0.1
    
    @pytest.mark.asyncio
    async def test_close_position(self, exchange_service, mock_exchange):
        # Arrange
        pair = "BTC/USDT:USDT"
        
        # Act
        result = await exchange_service.close_position(pair)
        
        # Assert
        mock_exchange.fetch_positions.assert_called_once_with(pair)
        mock_exchange.create_order.assert_called_once_with(
            pair, 'market', 'sell', 0.1, None, {"reduceOnly": True}
        )
        assert result["status"] == "closed"
        assert len(result["orders"]) == 1
    
    @pytest.mark.asyncio
    async def test_get_funding_rate(self, exchange_service, mock_exchange):
        # Arrange
        pair = "BTC/USDT:USDT"
        
        # Act
        result = await exchange_service.get_funding_rate(pair)
        
        # Assert
        mock_exchange.fetch_funding_rate.assert_called_once_with(pair)
        assert result["symbol"] == pair
        assert result["fundingRate"] == 0.0001
    
    @pytest.mark.asyncio
    async def test_get_contract_specifications(self, exchange_service, mock_exchange):
        # Arrange
        pair = "BTC/USDT:USDT"
        
        # Act
        result = await exchange_service.get_contract_specifications(pair)
        
        # Assert
        mock_exchange.fetch_market.assert_called_once_with(pair)
        assert result["pair"] == pair
        assert result["contract_size"] == 1
        assert result["price_precision"] == 0.5
        assert result["is_linear"] is True
        assert result["settlement_currency"] == "USDT"
    
    @pytest.mark.asyncio
    async def test_place_order_with_futures_initialization(self, exchange_service, mock_exchange):
        # Arrange
        pair = "BTC/USDT:USDT"
        order_type = "limit"
        order_side = "buy"
        amount = 0.1
        price = 50000
        params = {"reduceOnly": True}
        
        # Act
        result = await exchange_service.place_order(pair, order_type, order_side, amount, price, params)
        
        # Assert
        # Should initialize futures settings first
        mock_exchange.set_margin_mode.assert_called_once()
        mock_exchange.set_leverage.assert_called_once()
        
        # Then place the order
        mock_exchange.create_order.assert_called_once_with(pair, order_type, order_side, amount, price, params)
        assert result["id"] == "12345"
        assert result["symbol"] == pair
        assert result["status"] == "closed"