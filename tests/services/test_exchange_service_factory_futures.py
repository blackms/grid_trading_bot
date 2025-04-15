import pytest
from unittest.mock import patch, MagicMock
from config.config_manager import ConfigManager
from config.trading_mode import TradingMode
from config.market_type import MarketType
from core.services.exchange_service_factory import ExchangeServiceFactory
from core.services.backtest_exchange_service import BacktestExchangeService
from core.services.live_exchange_service import LiveExchangeService

class TestExchangeServiceFactoryFutures:
    
    @pytest.fixture
    def mock_config_manager(self):
        config_manager = MagicMock(spec=ConfigManager)
        config_manager.get_market_type.return_value = MarketType.FUTURES
        config_manager.get_exchange_name.return_value = "bybit"
        config_manager.get_margin_type.return_value = "isolated"
        config_manager.get_leverage.return_value = 3
        config_manager.is_hedge_mode_enabled.return_value = False
        return config_manager
    
    @patch('core.services.backtest_exchange_service.BacktestExchangeService')
    def test_create_backtest_exchange_service_with_futures(self, mock_backtest_service, mock_config_manager):
        # Arrange
        mock_backtest_instance = MagicMock(spec=BacktestExchangeService)
        mock_backtest_service.return_value = mock_backtest_instance
        
        # Act
        result = ExchangeServiceFactory.create_exchange_service(
            mock_config_manager, 
            TradingMode.BACKTEST
        )
        
        # Assert
        assert result == mock_backtest_instance
        mock_backtest_service.assert_called_once_with(mock_config_manager)
        mock_config_manager.get_market_type.assert_called_once()
    
    @patch('core.services.live_exchange_service.LiveExchangeService')
    def test_create_paper_trading_exchange_service_with_futures(self, mock_live_service, mock_config_manager):
        # Arrange
        mock_live_instance = MagicMock(spec=LiveExchangeService)
        mock_live_service.return_value = mock_live_instance
        
        # Act
        result = ExchangeServiceFactory.create_exchange_service(
            mock_config_manager, 
            TradingMode.PAPER_TRADING
        )
        
        # Assert
        assert result == mock_live_instance
        mock_live_service.assert_called_once_with(
            mock_config_manager, 
            is_paper_trading_activated=True
        )
        mock_config_manager.get_market_type.assert_called_once()
    
    @patch('core.services.live_exchange_service.LiveExchangeService')
    def test_create_live_exchange_service_with_futures(self, mock_live_service, mock_config_manager):
        # Arrange
        mock_live_instance = MagicMock(spec=LiveExchangeService)
        mock_live_service.return_value = mock_live_instance
        
        # Act
        result = ExchangeServiceFactory.create_exchange_service(
            mock_config_manager, 
            TradingMode.LIVE
        )
        
        # Assert
        assert result == mock_live_instance
        mock_live_service.assert_called_once_with(
            mock_config_manager, 
            is_paper_trading_activated=False
        )
        mock_config_manager.get_market_type.assert_called_once()
    
    def test_create_exchange_service_with_unsupported_trading_mode(self, mock_config_manager):
        # Arrange
        unsupported_trading_mode = "unsupported"
        
        # Act & Assert
        with pytest.raises(ValueError, match=f"Unsupported trading mode: {unsupported_trading_mode}"):
            ExchangeServiceFactory.create_exchange_service(
                mock_config_manager, 
                unsupported_trading_mode
            )