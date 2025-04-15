import pytest
import os
import json
from unittest.mock import MagicMock, patch

from config.config_manager import ConfigManager
from config.config_validator import ConfigValidator

class TestDynamicGridSettings:
    
    @pytest.fixture
    def sample_config_path(self, tmp_path):
        """Create a temporary config file with dynamic grid settings."""
        config_file = tmp_path / "test_config.json"
        
        config_data = {
            "exchange": {
                "name": "bybit",
                "trading_fee": 0.0006,
                "trading_mode": "live",
                "market_type": "futures"
            },
            "futures_settings": {
                "contract_type": "perpetual",
                "leverage": 3,
                "margin_type": "isolated",
                "hedge_mode": False
            },
            "pair": {
                "base_currency": "BTC",
                "quote_currency": "USDT",
                "contract_size": 1
            },
            "trading_settings": {
                "timeframe": "1h",
                "period": {
                    "start_date": "2025-01-01",
                    "end_date": "2025-04-15"
                },
                "initial_balance": 10000
            },
            "grid_strategy": {
                "type": "simple_grid",
                "spacing": "arithmetic",
                "num_grids": 10,
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
            },
            "risk_management": {
                "take_profit": {
                    "enabled": False,
                    "threshold": 0.05
                },
                "stop_loss": {
                    "enabled": False,
                    "threshold": 0.05
                },
                "futures": {
                    "liquidation_protection": {
                        "enabled": True,
                        "threshold": 0.5
                    },
                    "max_position_size": 0.5
                }
            },
            "logging": {
                "log_level": "INFO",
                "log_to_file": True
            }
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        return str(config_file)
    
    @pytest.fixture
    def config_manager(self, sample_config_path):
        """Create a ConfigManager instance with the sample config."""
        # Mock the validator to avoid validation errors
        validator = MagicMock(spec=ConfigValidator)
        validator.validate.return_value = None
        
        return ConfigManager(sample_config_path, validator)
    
    def test_is_dynamic_grid_enabled(self, config_manager):
        """Test that dynamic grid is correctly detected as enabled."""
        assert config_manager.is_dynamic_grid_enabled() is True
    
    def test_trailing_settings(self, config_manager):
        """Test that trailing settings are correctly retrieved."""
        assert config_manager.is_trailing_enabled() is True
        assert config_manager.get_trailing_activation_threshold() == 0.02
        assert config_manager.get_trailing_distance_percentage() == 0.01
        assert config_manager.get_trailing_cooldown_period() == 300
    
    def test_volatility_adaptation_settings(self, config_manager):
        """Test that volatility adaptation settings are correctly retrieved."""
        assert config_manager.is_volatility_adaptation_enabled() is True
        assert config_manager.get_volatility_lookback_period() == 24
        assert config_manager.get_volatility_grid_adjustment_factor() == 1.5
    
    def test_grid_repositioning_settings(self, config_manager):
        """Test that grid repositioning settings are correctly retrieved."""
        assert config_manager.is_grid_repositioning_enabled() is True
        assert config_manager.get_grid_repositioning_threshold() == 0.05
    
    def test_small_capital_optimization_settings(self, config_manager):
        """Test that small capital optimization settings are correctly retrieved."""
        assert config_manager.is_small_capital_optimization_enabled() is True
        assert config_manager.get_min_order_value() == 5.0
    
    def test_default_values(self):
        """Test that default values are returned when settings are not present."""
        # Create a config without dynamic grid settings
        config_data = {
            "exchange": {"name": "bybit", "trading_fee": 0.0006, "trading_mode": "live"},
            "pair": {"base_currency": "BTC", "quote_currency": "USDT"},
            "trading_settings": {
                "timeframe": "1h",
                "period": {"start_date": "2025-01-01", "end_date": "2025-04-15"},
                "initial_balance": 10000
            },
            "grid_strategy": {
                "type": "simple_grid",
                "spacing": "arithmetic",
                "num_grids": 10,
                "range": {"top": 22000.0, "bottom": 18000.0}
                # No dynamic_grid section
            },
            "risk_management": {
                "take_profit": {"enabled": False, "threshold": 0.05},
                "stop_loss": {"enabled": False, "threshold": 0.05}
            },
            "logging": {"log_level": "INFO", "log_to_file": True}
        }
        
        # Mock the config manager
        config_manager = MagicMock(spec=ConfigManager)
        config_manager.get_dynamic_grid_settings.return_value = {}
        
        # Test default values
        with patch.object(ConfigManager, 'get_dynamic_grid_settings', return_value={}):
            config_manager = ConfigManager.__new__(ConfigManager)
            config_manager.get_dynamic_grid_settings = lambda: {}
            
            assert config_manager.is_dynamic_grid_enabled() is False
            assert config_manager.is_trailing_enabled() is False
            assert config_manager.get_trailing_activation_threshold() == 0.02
            assert config_manager.get_trailing_distance_percentage() == 0.01
            assert config_manager.get_trailing_cooldown_period() == 300
            assert config_manager.is_volatility_adaptation_enabled() is False
            assert config_manager.get_volatility_lookback_period() == 24
            assert config_manager.get_volatility_grid_adjustment_factor() == 1.5
            assert config_manager.is_grid_repositioning_enabled() is False
            assert config_manager.get_grid_repositioning_threshold() == 0.05
            assert config_manager.is_small_capital_optimization_enabled() is False
            assert config_manager.get_min_order_value() == 5.0