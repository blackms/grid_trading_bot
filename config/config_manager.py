import json, os, logging
from typing import Optional
from strategies.spacing_type import SpacingType
from strategies.strategy_type import StrategyType
from .trading_mode import TradingMode
from .market_type import MarketType
from .exceptions import ConfigFileNotFoundError, ConfigParseError

class ConfigManager:
    def __init__(self, config_file, config_validator):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config_file = config_file
        self.config_validator = config_validator
        self.config = None
        self.load_config()

    def load_config(self):
        if not os.path.exists(self.config_file):
            self.logger.error(f"Config file {self.config_file} does not exist.")
            raise ConfigFileNotFoundError(self.config_file)
        
        with open(self.config_file, 'r') as file:
            try:
                self.config = json.load(file)
                self.config_validator.validate(self.config)
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse config file {self.config_file}: {e}")
                raise ConfigParseError(self.config_file, e)

    def get(self, key, default=None):
        return self.config.get(key, default)

    # --- General Accessor Methods ---
    def get_exchange(self):
        return self.config.get('exchange', {})
    
    def get_exchange_name(self):
        exchange = self.get_exchange()
        return exchange.get('name', None)

    def get_trading_fee(self):
        exchange = self.get_exchange()
        return exchange.get('trading_fee', 0)
    
    def get_trading_mode(self) -> Optional[TradingMode]:
        exchange = self.get_exchange()
        trading_mode = exchange.get('trading_mode', None)
        
        if trading_mode:
            return TradingMode.from_string(trading_mode)

    def get_pair(self):
        return self.config.get('pair', {})
    
    def get_base_currency(self):
        pair = self.get_pair()
        return pair.get('base_currency', None)

    def get_quote_currency(self):
        pair = self.get_pair()
        return pair.get('quote_currency', None)
    
    def get_trading_settings(self):
        return self.config.get('trading_settings', {})

    def get_timeframe(self):
        trading_settings = self.get_trading_settings()
        return trading_settings.get('timeframe', '1h')

    def get_period(self):
        trading_settings = self.get_trading_settings()
        return trading_settings.get('period', {})
    
    def get_start_date(self):
        period = self.get_period()
        return period.get('start_date', None)

    def get_end_date(self):
        period = self.get_period()
        return period.get('end_date', None)

    def get_initial_balance(self):
        trading_settings = self.get_trading_settings()
        return trading_settings.get('initial_balance', 10000)
    
    def get_historical_data_file(self):
        trading_settings = self.get_trading_settings()
        return trading_settings.get('historical_data_file', None)

    # --- Grid Accessor Methods ---
    def get_grid_settings(self):
        return self.config.get('grid_strategy', {})

    def get_strategy_type(self) -> Optional[StrategyType]:
        grid_settings = self.get_grid_settings()
        strategy_type = grid_settings.get('type', None)

        if strategy_type:
            return StrategyType.from_string(strategy_type)
    
    def get_spacing_type(self)-> Optional[SpacingType]:
        grid_settings = self.get_grid_settings()
        spacing_type = grid_settings.get('spacing', None)
    
        if spacing_type:
            return SpacingType.from_string(spacing_type)

    def get_num_grids(self):
        grid_settings = self.get_grid_settings()
        return grid_settings.get('num_grids', None)
    
    def get_grid_range(self):
        grid_settings = self.get_grid_settings()
        return grid_settings.get('range', {})

    def get_top_range(self):
        grid_range = self.get_grid_range()
        return grid_range.get('top', None)

    def get_bottom_range(self):
        grid_range = self.get_grid_range()
        return grid_range.get('bottom', None)

    # --- Risk management (Take Profit / Stop Loss) Accessor Methods ---
    def get_risk_management(self):
        return self.config.get('risk_management', {})

    def get_take_profit(self):
        risk_management = self.get_risk_management()
        return risk_management.get('take_profit', {})

    def is_take_profit_enabled(self):
        take_profit = self.get_take_profit()
        return take_profit.get('enabled', False)

    def get_take_profit_threshold(self):
        take_profit = self.get_take_profit()
        return take_profit.get('threshold', None)

    def get_stop_loss(self):
        risk_management = self.get_risk_management()
        return risk_management.get('stop_loss', {})

    def is_stop_loss_enabled(self):
        stop_loss = self.get_stop_loss()
        return stop_loss.get('enabled', False)

    def get_stop_loss_threshold(self):
        stop_loss = self.get_stop_loss()
        return stop_loss.get('threshold', None)

    # --- Market Type Accessor Methods ---
    def get_market_type(self) -> Optional[MarketType]:
        exchange = self.get_exchange()
        market_type = exchange.get('market_type', 'spot')  # Default to spot for backward compatibility
        
        if market_type:
            return MarketType.from_string(market_type)
    
    def is_futures_market(self) -> bool:
        market_type = self.get_market_type()
        return market_type == MarketType.FUTURES if market_type else False
    
    # --- Futures Settings Accessor Methods ---
    def get_futures_settings(self):
        return self.config.get('futures_settings', {})
    
    def get_contract_type(self):
        futures_settings = self.get_futures_settings()
        return futures_settings.get('contract_type', 'perpetual')
    
    def get_leverage(self):
        futures_settings = self.get_futures_settings()
        return futures_settings.get('leverage', 1)
    
    def get_margin_type(self):
        futures_settings = self.get_futures_settings()
        return futures_settings.get('margin_type', 'isolated')
    
    def is_hedge_mode_enabled(self) -> bool:
        futures_settings = self.get_futures_settings()
        return futures_settings.get('hedge_mode', False)
    
    def get_contract_size(self):
        pair = self.get_pair()
        return pair.get('contract_size', 1)
    
    # --- Futures Risk Management Accessor Methods ---
    def get_futures_risk_management(self):
        risk_management = self.get_risk_management()
        return risk_management.get('futures', {})
    
    def get_liquidation_protection(self):
        futures_risk = self.get_futures_risk_management()
        return futures_risk.get('liquidation_protection', {})
    
    def is_liquidation_protection_enabled(self) -> bool:
        liquidation_protection = self.get_liquidation_protection()
        return liquidation_protection.get('enabled', False)
    
    def get_liquidation_protection_threshold(self):
        liquidation_protection = self.get_liquidation_protection()
        return liquidation_protection.get('threshold', 0.5)  # Default to 50% of the distance to liquidation
    
    def get_max_position_size(self):
        futures_risk = self.get_futures_risk_management()
        return futures_risk.get('max_position_size', None)
    
    # --- Funding Rate Settings Accessor Methods ---
    def get_funding_rate_settings(self):
        futures_risk = self.get_futures_risk_management()
        return futures_risk.get('funding_rate', {})
    
    def get_funding_rate_monitoring_settings(self):
        funding_rate = self.get_funding_rate_settings()
        return funding_rate.get('monitoring', {})
    
    def is_funding_rate_monitoring_enabled(self) -> bool:
        monitoring = self.get_funding_rate_monitoring_settings()
        return monitoring.get('enabled', False)
    
    def get_funding_rate_update_interval(self) -> int:
        monitoring = self.get_funding_rate_monitoring_settings()
        return monitoring.get('update_interval', 900)  # Default to 15 minutes
    
    def get_funding_rate_notification_threshold(self) -> int:
        monitoring = self.get_funding_rate_monitoring_settings()
        return monitoring.get('notification_threshold', 1800)  # Default to 30 minutes
    
    def get_funding_rate_thresholds(self):
        funding_rate = self.get_funding_rate_settings()
        return funding_rate.get('thresholds', {})
    
    def get_funding_rate_high_threshold(self) -> float:
        thresholds = self.get_funding_rate_thresholds()
        return thresholds.get('high', 0.001)  # Default to 0.1%
    
    def get_funding_rate_extreme_threshold(self) -> float:
        thresholds = self.get_funding_rate_thresholds()
        return thresholds.get('extreme', 0.003)  # Default to 0.3%
    
    def get_funding_rate_cumulative_threshold(self) -> float:
        thresholds = self.get_funding_rate_thresholds()
        return thresholds.get('cumulative', 0.01)  # Default to 1%
    
    def get_funding_rate_strategy_adjustment_settings(self):
        funding_rate = self.get_funding_rate_settings()
        return funding_rate.get('strategy_adjustments', {})
    
    def is_funding_rate_strategy_adjustment_enabled(self) -> bool:
        adjustments = self.get_funding_rate_strategy_adjustment_settings()
        return adjustments.get('enabled', False)
    
    def get_funding_rate_reduce_exposure_threshold(self) -> float:
        adjustments = self.get_funding_rate_strategy_adjustment_settings()
        return adjustments.get('reduce_exposure_threshold', 0.002)  # Default to 0.2%
    
    def get_funding_rate_reverse_position_threshold(self) -> float:
        adjustments = self.get_funding_rate_strategy_adjustment_settings()
        return adjustments.get('reverse_position_threshold', 0.005)  # Default to 0.5%
    
    def get_funding_rate_max_adjustment_percentage(self) -> float:
        adjustments = self.get_funding_rate_strategy_adjustment_settings()
        return adjustments.get('max_adjustment_percentage', 0.5)  # Default to 50%
    
    def get_funding_rate_forecasting_settings(self):
        funding_rate = self.get_funding_rate_settings()
        return funding_rate.get('forecasting', {})
    
    def is_funding_rate_forecasting_enabled(self) -> bool:
        forecasting = self.get_funding_rate_forecasting_settings()
        return forecasting.get('enabled', False)
    
    def get_funding_rate_forecast_window(self) -> int:
        forecasting = self.get_funding_rate_forecasting_settings()
        return forecasting.get('forecast_window', 3)  # Default to 3 periods
    
    def get_funding_rate_min_history_periods(self) -> int:
        forecasting = self.get_funding_rate_forecasting_settings()
        return forecasting.get('min_history_periods', 6)  # Default to 6 periods
    
    def get_funding_rate_confidence_threshold(self) -> float:
        forecasting = self.get_funding_rate_forecasting_settings()
        return forecasting.get('confidence_threshold', 0.7)  # Default to 70%
    
    def get_funding_rate_auto_hedge_settings(self):
        funding_rate = self.get_funding_rate_settings()
        return funding_rate.get('auto_hedge', {})
    
    def is_funding_rate_auto_hedge_enabled(self) -> bool:
        auto_hedge = self.get_funding_rate_auto_hedge_settings()
        return auto_hedge.get('enabled', False)
    
    def get_funding_rate_hedge_threshold(self) -> float:
        auto_hedge = self.get_funding_rate_auto_hedge_settings()
        return auto_hedge.get('hedge_threshold', 0.003)  # Default to 0.3%
    
    def get_funding_rate_max_hedge_ratio(self) -> float:
        auto_hedge = self.get_funding_rate_auto_hedge_settings()
        return auto_hedge.get('max_hedge_ratio', 0.5)  # Default to 50%
    
    def get_funding_rate_min_funding_duration(self) -> int:
        auto_hedge = self.get_funding_rate_auto_hedge_settings()
        return auto_hedge.get('min_funding_duration', 24)  # Default to 24 hours
    
    # --- Logging Accessor Methods ---
    def get_logging(self):
        return self.config.get('logging', {})
    
    def get_logging_level(self):
        logging = self.get_logging()
        return logging.get('log_level', {})
    
    def should_log_to_file(self) -> bool:
        logging = self.get_logging()
        return logging.get('log_to_file', False)
    
    # --- Dynamic Grid Settings Accessor Methods ---
    def get_dynamic_grid_settings(self):
        grid_settings = self.get_grid_settings()
        return grid_settings.get('dynamic_grid', {})
    
    def is_dynamic_grid_enabled(self) -> bool:
        dynamic_grid = self.get_dynamic_grid_settings()
        return bool(dynamic_grid)  # Return True if dynamic_grid is not empty
    
    def is_trailing_enabled(self) -> bool:
        dynamic_grid = self.get_dynamic_grid_settings()
        return dynamic_grid.get('trailing_enabled', False)
    
    def get_trailing_activation_threshold(self) -> float:
        dynamic_grid = self.get_dynamic_grid_settings()
        return dynamic_grid.get('trailing_activation_threshold', 0.02)  # Default to 2%
    
    def get_trailing_distance_percentage(self) -> float:
        dynamic_grid = self.get_dynamic_grid_settings()
        return dynamic_grid.get('trailing_distance_percentage', 0.01)  # Default to 1%
    
    def get_trailing_cooldown_period(self) -> int:
        dynamic_grid = self.get_dynamic_grid_settings()
        return dynamic_grid.get('trailing_cooldown_period', 300)  # Default to 5 minutes
    
    def is_volatility_adaptation_enabled(self) -> bool:
        dynamic_grid = self.get_dynamic_grid_settings()
        return dynamic_grid.get('volatility_adaptation_enabled', False)
    
    def get_volatility_lookback_period(self) -> int:
        dynamic_grid = self.get_dynamic_grid_settings()
        return dynamic_grid.get('volatility_lookback_period', 24)  # Default to 24 hours
    
    def get_volatility_grid_adjustment_factor(self) -> float:
        dynamic_grid = self.get_dynamic_grid_settings()
        return dynamic_grid.get('volatility_grid_adjustment_factor', 1.5)  # Default to 1.5x
    
    def is_grid_repositioning_enabled(self) -> bool:
        dynamic_grid = self.get_dynamic_grid_settings()
        return dynamic_grid.get('grid_repositioning_enabled', False)
    
    def get_grid_repositioning_threshold(self) -> float:
        dynamic_grid = self.get_dynamic_grid_settings()
        return dynamic_grid.get('grid_repositioning_threshold', 0.05)  # Default to 5%
    
    def is_small_capital_optimization_enabled(self) -> bool:
        dynamic_grid = self.get_dynamic_grid_settings()
        return dynamic_grid.get('small_capital_optimization_enabled', False)
    
    def get_min_order_value(self) -> float:
        dynamic_grid = self.get_dynamic_grid_settings()
        return dynamic_grid.get('min_order_value', 5.0)  # Default to 5 units of quote currency
        
    # --- Stop Loss Manager Settings Accessor Methods ---
    def get_stop_loss_manager_settings(self):
        futures_risk = self.get_futures_risk_management()
        return futures_risk.get('stop_loss_manager', {})
    
    def is_stop_loss_manager_enabled(self) -> bool:
        stop_loss_manager = self.get_stop_loss_manager_settings()
        return stop_loss_manager.get('enabled', True)  # Default to enabled for futures
    
    def get_stop_loss_monitoring_interval(self) -> int:
        stop_loss_manager = self.get_stop_loss_manager_settings()
        return stop_loss_manager.get('monitoring_interval', 5)  # Default to 5 seconds
    
    def get_stop_loss_execution_type(self) -> str:
        stop_loss_manager = self.get_stop_loss_manager_settings()
        return stop_loss_manager.get('execution_type', 'market')  # Default to market order
    
    def is_partial_close_enabled(self) -> bool:
        stop_loss_manager = self.get_stop_loss_manager_settings()
        return stop_loss_manager.get('partial_close_enabled', False)
    
    def get_partial_close_percentage(self) -> float:
        stop_loss_manager = self.get_stop_loss_manager_settings()
        return stop_loss_manager.get('partial_close_percentage', 0.5)  # Default to 50%
    
    def get_usdt_stop_loss_settings(self):
        stop_loss_manager = self.get_stop_loss_manager_settings()
        return stop_loss_manager.get('usdt_stop_loss', {})
    
    def is_usdt_stop_loss_enabled(self) -> bool:
        usdt_stop_loss = self.get_usdt_stop_loss_settings()
        return usdt_stop_loss.get('enabled', False)
    
    def get_usdt_stop_loss_max_loss_amount(self) -> float:
        usdt_stop_loss = self.get_usdt_stop_loss_settings()
        return usdt_stop_loss.get('max_loss_amount', 1000.0)  # Default to 1000 USDT
    
    def is_usdt_stop_loss_per_position(self) -> bool:
        usdt_stop_loss = self.get_usdt_stop_loss_settings()
        return usdt_stop_loss.get('per_position', False)
    
    def get_usdt_stop_loss_warning_threshold(self) -> float:
        usdt_stop_loss = self.get_usdt_stop_loss_settings()
        return usdt_stop_loss.get('warning_threshold', 0.7)  # Default to 70% of max loss
    
    def get_portfolio_stop_loss_settings(self):
        stop_loss_manager = self.get_stop_loss_manager_settings()
        return stop_loss_manager.get('portfolio_stop_loss', {})
    
    def is_portfolio_stop_loss_enabled(self) -> bool:
        portfolio_stop_loss = self.get_portfolio_stop_loss_settings()
        return portfolio_stop_loss.get('enabled', False)
    
    def get_portfolio_stop_loss_max_loss_percentage(self) -> float:
        portfolio_stop_loss = self.get_portfolio_stop_loss_settings()
        return portfolio_stop_loss.get('max_loss_percentage', 0.1)  # Default to 10% of portfolio
    
    def get_portfolio_stop_loss_warning_threshold(self) -> float:
        portfolio_stop_loss = self.get_portfolio_stop_loss_settings()
        return portfolio_stop_loss.get('warning_threshold', 0.7)  # Default to 70% of max loss
    
    def get_trailing_stop_loss_settings(self):
        stop_loss_manager = self.get_stop_loss_manager_settings()
        return stop_loss_manager.get('trailing_stop_loss', {})
    
    def is_trailing_stop_loss_enabled(self) -> bool:
        trailing_stop_loss = self.get_trailing_stop_loss_settings()
        return trailing_stop_loss.get('enabled', False)
    
    def get_trailing_stop_loss_activation_threshold(self) -> float:
        trailing_stop_loss = self.get_trailing_stop_loss_settings()
        return trailing_stop_loss.get('activation_threshold', 0.02)  # Default to 2% profit
    
    def get_trailing_stop_loss_distance(self) -> float:
        trailing_stop_loss = self.get_trailing_stop_loss_settings()
        return trailing_stop_loss.get('trailing_distance', 0.01)  # Default to 1% trailing distance
    
    def get_external_signals_settings(self):
        stop_loss_manager = self.get_stop_loss_manager_settings()
        return stop_loss_manager.get('external_signals', {})
    
    def is_external_signals_enabled(self) -> bool:
        external_signals = self.get_external_signals_settings()
        return external_signals.get('enabled', False)
    
    def get_external_signals_sources(self) -> list:
        external_signals = self.get_external_signals_settings()
        return external_signals.get('signal_sources', [])