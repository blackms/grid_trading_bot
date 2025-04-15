import pytest, json
from unittest.mock import patch, mock_open, Mock
from config.config_manager import ConfigManager
from config.config_validator import ConfigValidator
from strategies.spacing_type import SpacingType
from strategies.strategy_type import StrategyType
from config.trading_mode import TradingMode
from config.market_type import MarketType
from config.exceptions import ConfigFileNotFoundError, ConfigParseError

class TestConfigManager:
    @pytest.fixture
    def mock_validator(self):
        return Mock(spec=ConfigValidator)

    @pytest.fixture
    def config_manager(self, mock_validator, valid_config):
        # Mocking both open and os.path.exists to simulate a valid config file
        mocked_open = mock_open(read_data=json.dumps(valid_config))
        with patch("builtins.open", mocked_open), patch("os.path.exists", return_value=True):
            return ConfigManager("config.json", mock_validator)

    def test_load_config_valid(self, config_manager, valid_config, mock_validator):
        mock_validator.validate.assert_called_once_with(valid_config)
        assert config_manager.config == valid_config

    def test_load_config_file_not_found(self, mock_validator):
        with patch("os.path.exists", return_value=False):
            with pytest.raises(ConfigFileNotFoundError):
                ConfigManager("config.json", mock_validator)

    def test_load_config_json_decode_error(self, mock_validator):
        invalid_json = '{"invalid_json": '  # Malformed JSON
        mocked_open = mock_open(read_data=invalid_json)
        with patch("builtins.open", mocked_open), patch("os.path.exists", return_value=True):
            with pytest.raises(ConfigParseError):
                ConfigManager("config.json", mock_validator)

    def test_get_exchange_name(self, config_manager):
        assert config_manager.get_exchange_name() == "binance"

    def test_get_trading_fee(self, config_manager):
        assert config_manager.get_trading_fee() == 0.001

    def test_get_base_currency(self, config_manager):
        assert config_manager.get_base_currency() == "ETH"

    def test_get_quote_currency(self, config_manager):
        assert config_manager.get_quote_currency() == "USDT"

    def test_get_initial_balance(self, config_manager):
        assert config_manager.get_initial_balance() == 10000
    
    def test_get_spacing_type(self, config_manager):
        assert config_manager.get_spacing_type() == SpacingType.GEOMETRIC

    def test_get_strategy_type(self, config_manager):
        assert config_manager.get_strategy_type() == StrategyType.SIMPLE_GRID

    def test_get_trading_mode(self, config_manager):
        assert config_manager.get_trading_mode() == TradingMode.BACKTEST

    def test_get_timeframe(self, config_manager):
        assert config_manager.get_timeframe() == "1m"

    def test_get_period(self, config_manager):
        expected_period = {
            "start_date": "2024-07-04T00:00:00Z",
            "end_date": "2024-07-11T00:00:00Z"
        }
        assert config_manager.get_period() == expected_period
    
    def test_get_start_date(self, config_manager):
        assert config_manager.get_start_date() == "2024-07-04T00:00:00Z"
    
    def test_get_end_date(self, config_manager):
        assert config_manager.get_end_date() == "2024-07-11T00:00:00Z"

    def test_get_num_grids(self, config_manager):
        assert config_manager.get_num_grids() == 20

    def test_get_grid_range(self, config_manager):
        expected_range = {
            "top": 3100,
            "bottom": 2850
        }
        assert config_manager.get_grid_range() == expected_range
    
    def test_get_top_range(self, config_manager):
        assert config_manager.get_top_range() == 3100
    
    def test_get_bottom_range(self, config_manager):
        assert config_manager.get_bottom_range() == 2850

    def test_is_take_profit_enabled(self, config_manager):
        assert config_manager.is_take_profit_enabled() == False

    def test_get_take_profit_threshold(self, config_manager):
        assert config_manager.get_take_profit_threshold() == 3700

    def test_get_stop_loss_threshold(self, config_manager):
        assert config_manager.get_stop_loss_threshold() == 2830

    def test_is_stop_loss_enabled(self, config_manager):
        assert config_manager.is_stop_loss_enabled() == False

    def test_get_log_level(self, config_manager):
        assert config_manager.get_logging_level() == "INFO"

    def test_should_log_to_file_true(self, config_manager):
        assert config_manager.should_log_to_file() is True
    
    def test_get_trading_mode_invalid_value(self, config_manager):
        config_manager.config["exchange"]["trading_mode"] = "invalid_mode"
        
        with pytest.raises(ValueError, match="Invalid trading mode: 'invalid_mode'. Available modes are: backtest, paper_trading, live"):
            config_manager.get_trading_mode()

    def test_get_spacing_type_invalid_value(self, config_manager):
        config_manager.config["grid_strategy"]["spacing"] = "invalid_spacing"
        
        with pytest.raises(ValueError, match="Invalid spacing type: 'invalid_spacing'. Available spacings are: arithmetic, geometric"):
            config_manager.get_spacing_type()
        
    def test_get_strategy_type_invalid_value(self, config_manager):
        config_manager.config["grid_strategy"]["type"] = "invalid_strategy"
        
        with pytest.raises(ValueError, match="Invalid strategy type: 'invalid_strategy'. Available strategies are: simple_grid, hedged_grid"):
            config_manager.get_strategy_type()
    
    def test_get_timeframe_default(self, config_manager):
        del config_manager.config["trading_settings"]["timeframe"]
        assert config_manager.get_timeframe() == "1h"

    def test_get_historical_data_file_default(self, config_manager):
        del config_manager.config["trading_settings"]["historical_data_file"]
        assert config_manager.get_historical_data_file() is None
    
    def test_is_take_profit_enabled_default(self, config_manager):
        del config_manager.config["risk_management"]["take_profit"]
        assert config_manager.is_take_profit_enabled() is False

    def test_get_take_profit_threshold_default(self, config_manager):
        del config_manager.config["risk_management"]["take_profit"]
        assert config_manager.get_take_profit_threshold() is None

    def test_is_stop_loss_enabled_default(self, config_manager):
        del config_manager.config["risk_management"]["stop_loss"]
        assert config_manager.is_stop_loss_enabled() is False

    def test_get_stop_loss_threshold_default(self, config_manager):
        del config_manager.config["risk_management"]["stop_loss"]
        assert config_manager.get_stop_loss_threshold() is None
    
    # --- Market Type Tests ---
    def test_get_market_type_spot(self, config_manager):
        config_manager.config["exchange"]["market_type"] = "spot"
        assert config_manager.get_market_type() == MarketType.SPOT
    
    def test_get_market_type_futures(self, config_manager):
        config_manager.config["exchange"]["market_type"] = "futures"
        assert config_manager.get_market_type() == MarketType.FUTURES
    
    def test_get_market_type_default(self, config_manager):
        # Default should be spot for backward compatibility
        assert config_manager.get_market_type() == MarketType.SPOT
    
    def test_is_futures_market_true(self, config_manager):
        config_manager.config["exchange"]["market_type"] = "futures"
        assert config_manager.is_futures_market() is True
    
    def test_is_futures_market_false(self, config_manager):
        config_manager.config["exchange"]["market_type"] = "spot"
        assert config_manager.is_futures_market() is False
    
    def test_get_market_type_invalid_value(self, config_manager):
        config_manager.config["exchange"]["market_type"] = "invalid_market"
        
        with pytest.raises(ValueError, match="Invalid market type: 'invalid_market'. Available market types are: spot, futures"):
            config_manager.get_market_type()
    
    # --- Futures Settings Tests ---
    def test_get_futures_settings(self, config_manager):
        config_manager.config["futures_settings"] = {
            "contract_type": "perpetual",
            "leverage": 5,
            "margin_type": "isolated",
            "hedge_mode": True
        }
        assert config_manager.get_futures_settings() == {
            "contract_type": "perpetual",
            "leverage": 5,
            "margin_type": "isolated",
            "hedge_mode": True
        }
    
    def test_get_contract_type(self, config_manager):
        config_manager.config["futures_settings"] = {"contract_type": "perpetual"}
        assert config_manager.get_contract_type() == "perpetual"
    
    def test_get_contract_type_default(self, config_manager):
        # No futures_settings in config
        assert config_manager.get_contract_type() == "perpetual"
    
    def test_get_leverage(self, config_manager):
        config_manager.config["futures_settings"] = {"leverage": 5}
        assert config_manager.get_leverage() == 5
    
    def test_get_leverage_default(self, config_manager):
        # No futures_settings in config
        assert config_manager.get_leverage() == 1
    
    def test_get_margin_type(self, config_manager):
        config_manager.config["futures_settings"] = {"margin_type": "cross"}
        assert config_manager.get_margin_type() == "cross"
    
    def test_get_margin_type_default(self, config_manager):
        # No futures_settings in config
        assert config_manager.get_margin_type() == "isolated"
    
    def test_is_hedge_mode_enabled(self, config_manager):
        config_manager.config["futures_settings"] = {"hedge_mode": True}
        assert config_manager.is_hedge_mode_enabled() is True
    
    def test_is_hedge_mode_enabled_default(self, config_manager):
        # No futures_settings in config
        assert config_manager.is_hedge_mode_enabled() is False
    
    def test_get_contract_size(self, config_manager):
        config_manager.config["pair"]["contract_size"] = 0.01
        assert config_manager.get_contract_size() == 0.01
    
    def test_get_contract_size_default(self, config_manager):
        # No contract_size in pair
        assert config_manager.get_contract_size() == 1
    
    # --- Futures Risk Management Tests ---
    def test_get_futures_risk_management(self, config_manager):
        config_manager.config["risk_management"]["futures"] = {
            "liquidation_protection": {
                "enabled": True,
                "threshold": 0.7
            },
            "max_position_size": 10
        }
        expected = {
            "liquidation_protection": {
                "enabled": True,
                "threshold": 0.7
            },
            "max_position_size": 10
        }
        assert config_manager.get_futures_risk_management() == expected
    
    def test_get_liquidation_protection(self, config_manager):
        config_manager.config["risk_management"]["futures"] = {
            "liquidation_protection": {
                "enabled": True,
                "threshold": 0.7
            }
        }
        expected = {
            "enabled": True,
            "threshold": 0.7
        }
        assert config_manager.get_liquidation_protection() == expected
    
    def test_is_liquidation_protection_enabled(self, config_manager):
        config_manager.config["risk_management"]["futures"] = {
            "liquidation_protection": {
                "enabled": True
            }
        }
        assert config_manager.is_liquidation_protection_enabled() is True
    
    def test_is_liquidation_protection_enabled_default(self, config_manager):
        # No futures risk management in config
        assert config_manager.is_liquidation_protection_enabled() is False
    
    def test_get_liquidation_protection_threshold(self, config_manager):
        config_manager.config["risk_management"]["futures"] = {
            "liquidation_protection": {
                "threshold": 0.7
            }
        }
        assert config_manager.get_liquidation_protection_threshold() == 0.7
    
    def test_get_liquidation_protection_threshold_default(self, config_manager):
        # No futures risk management in config
        assert config_manager.get_liquidation_protection_threshold() == 0.5
    
    def test_get_max_position_size(self, config_manager):
        config_manager.config["risk_management"]["futures"] = {
            "max_position_size": 10
        }
        assert config_manager.get_max_position_size() == 10
    
    def test_get_max_position_size_default(self, config_manager):
        # No futures risk management in config
        assert config_manager.get_max_position_size() is None