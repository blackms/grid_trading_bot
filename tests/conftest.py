import pytest

@pytest.fixture
def valid_config():
    """Fixture providing a valid configuration for testing."""
    return {
        "exchange": {
            "name": "binance",
            "trading_fee": 0.001,
            'trading_mode': 'backtest',
            'market_type': 'spot'  # Default to spot for backward compatibility
        },
        "pair": {
            "base_currency": "ETH",
            "quote_currency": "USDT"
        },
        "futures_settings": {  # New section for futures settings
            "contract_type": "perpetual",
            "leverage": 3,
            "margin_type": "isolated",
            "hedge_mode": False
        },
        "trading_settings": {
            "initial_balance": 10000,
            "timeframe": "1m",
            "period": {
                "start_date": "2024-07-04T00:00:00Z",
                "end_date": "2024-07-11T00:00:00Z"
            },
            "historical_data_file": "data/SOL_USDT/2024/1m.csv"
        },
        "grid_strategy": {
            "type": "simple_grid",
            "spacing": "geometric",
            "num_grids": 20,
            "range": {
                "top": 3100,
                "bottom": 2850
            }
        },
        "risk_management": {
            "take_profit": {
                "enabled": False,
                "threshold": 3700
            },
            "stop_loss": {
                "enabled": False,
                "threshold": 2830
            },
            "futures": {  # New section for futures risk management
                "liquidation_protection": {
                    "enabled": True,
                    "threshold": 0.5
                },
                "max_position_size": 5
            }
        },
        "logging": {
            "log_level": "INFO",
            "log_to_file": True
        }
    }