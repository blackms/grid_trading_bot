import ccxt, logging, time, os
from typing import Optional, Dict, Any, Union, List
import pandas as pd
from config.config_manager import ConfigManager
from config.market_type import MarketType
from utils.constants import CANDLE_LIMITS, TIMEFRAME_MAPPINGS
from .exchange_interface import ExchangeInterface
from .exceptions import UnsupportedExchangeError, DataFetchError, UnsupportedTimeframeError, HistoricalMarketDataFileNotFoundError, UnsupportedPairError

class BacktestExchangeService(ExchangeInterface):
    def __init__(self, config_manager: ConfigManager):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config_manager = config_manager
        self.historical_data_file = self.config_manager.get_historical_data_file()
        self.exchange_name = self.config_manager.get_exchange_name()
        self.market_type = self.config_manager.get_market_type()
        self.exchange = self._initialize_exchange()
        
        # Initialize futures settings if applicable
        if self.market_type == MarketType.FUTURES:
            self._initialize_futures_settings()
    
    def _initialize_exchange(self) -> Optional[ccxt.Exchange]:
        try:
            options = {}
            
            # Add futures-specific options if needed
            if self.market_type == MarketType.FUTURES:
                options['options'] = {
                    'defaultType': 'future',
                    'marginType': self.config_manager.get_margin_type(),
                    'hedgeMode': self.config_manager.is_hedge_mode_enabled()
                }
                
            exchange = getattr(ccxt, self.exchange_name)(options)
            return exchange
        except AttributeError:
            raise UnsupportedExchangeError(f"The exchange '{self.exchange_name}' is not supported.")
            
    def _initialize_futures_settings(self) -> None:
        """Initialize futures-specific settings for backtesting."""
        self.logger.info("Initializing futures settings for backtesting")
        self.leverage = self.config_manager.get_leverage()
        self.margin_type = self.config_manager.get_margin_type()
        self.hedge_mode = self.config_manager.is_hedge_mode_enabled()
        self.contract_size = self.config_manager.get_contract_size()
        
        # Store simulated positions for backtesting
        self.positions = {}
    
    def _is_timeframe_supported(self, timeframe: str) -> bool:
        if timeframe not in self.exchange.timeframes:
            self.logger.error(f"Timeframe '{timeframe}' is not supported by {self.exchange_name}.")
            return False
        return True
    
    def _is_pair_supported(self, pair: str) -> bool:
        markets = self.exchange.load_markets()
        return pair in markets

    def fetch_ohlcv(
        self, 
        pair: str, 
        timeframe: str, 
        start_date: str, 
        end_date: str
    ) -> pd.DataFrame:
        if self.historical_data_file:
            if not os.path.exists(self.historical_data_file):
                raise HistoricalMarketDataFileNotFoundError(f"Failed to load OHLCV data from file: {self.historical_data_file}")
    
            self.logger.info(f"Loading OHLCV data from file: {self.historical_data_file}")
            return self._load_ohlcv_from_file(self.historical_data_file, start_date, end_date)
        
        if not self._is_pair_supported(pair):
            raise UnsupportedPairError(f"Pair: {pair} is not supported by {self.exchange_name}")

        if not self._is_timeframe_supported(timeframe):
            raise UnsupportedTimeframeError(f"Timeframe '{timeframe}' is not supported by {self.exchange_name}.")

        self.logger.info(f"Fetching OHLCV data for {pair} from {start_date} to {end_date}")
        try:
            since = self.exchange.parse8601(start_date)
            until = self.exchange.parse8601(end_date)
            candles_per_request = self._get_candle_limit()
            total_candles_needed = (until - since) // self._get_timeframe_in_ms(timeframe)

            if total_candles_needed > candles_per_request:
                return self._fetch_ohlcv_in_chunks(pair, timeframe, since, until, candles_per_request)
            else:
                return self._fetch_ohlcv_single_batch(pair, timeframe, since, until)
        except ccxt.NetworkError as e:
            raise DataFetchError(f"Network issue occurred while fetching OHLCV data: {str(e)}")
        except ccxt.BaseError as e:
            raise DataFetchError(f"Exchange-specific error occurred: {str(e)}")
        except Exception as e:
            raise DataFetchError(f"Failed to fetch OHLCV data {str(e)}.")
    
    def _load_ohlcv_from_file(
        self, 
        file_path: str, 
        start_date: str, 
        end_date: str
    ) -> pd.DataFrame:
        try:
            df = pd.read_csv(file_path, parse_dates=['timestamp'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            start_timestamp = pd.to_datetime(start_date).tz_localize(None)
            end_timestamp = pd.to_datetime(end_date).tz_localize(None)
            filtered_df = df.loc[start_timestamp:end_timestamp]
            self.logger.debug(f"Loaded {len(filtered_df)} rows of OHLCV data from file.")
            return filtered_df
            
        except Exception as e:
            raise DataFetchError(f"Failed to load OHLCV data from file: {str(e)}")

    def _fetch_ohlcv_single_batch(
        self, 
        pair: str, 
        timeframe: str, 
        since: int, 
        until: int
    ) -> pd.DataFrame:
        ohlcv = self._fetch_with_retry(self.exchange.fetch_ohlcv, pair, timeframe, since)
        return self._format_ohlcv(ohlcv, until)

    def _fetch_ohlcv_in_chunks(
        self, 
        pair: str, 
        timeframe: str, 
        since: int,
        until: int,
        candles_per_request: int
    ) -> pd.DataFrame:
        all_ohlcv = []
        while since < until:
            ohlcv = self._fetch_with_retry(self.exchange.fetch_ohlcv, pair, timeframe, since, limit=candles_per_request)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            self.logger.info(f"Fetched up to {pd.to_datetime(since, unit='ms')}")
        return self._format_ohlcv(all_ohlcv, until)

    def _format_ohlcv(
        self, 
        ohlcv, 
        until: int
    ) -> pd.DataFrame:
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        until_timestamp = pd.to_datetime(until, unit='ms')
        return df[df.index <= until_timestamp]
    
    def _get_candle_limit(self) -> int:
        return CANDLE_LIMITS.get(self.exchange_name, 500)  # Default to 500 if not found

    def _get_timeframe_in_ms(self, timeframe: str) -> int:
        return TIMEFRAME_MAPPINGS.get(timeframe, 60 * 1000)  # Default to 1m if not found

    def _fetch_with_retry(
        self,
        method, 
        *args, 
        retries=3, 
        delay=5, 
        **kwargs
    ):
        for attempt in range(retries):
            try:
                return method(*args, **kwargs)
            except Exception as e:
                if attempt < retries - 1:
                    self.logger.warning(f"Attempt {attempt+1} failed. Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    self.logger.error(f"Failed after {retries} attempts: {e}")
                    raise DataFetchError(f"Failed to fetch data after {retries} attempts: {str(e)}")

    async def place_order(
        self,
        pair: str,
        order_side: str,
        order_type: str,
        amount: float,
        price: Optional[float] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Union[str, float]]:
        raise NotImplementedError("place_order is not used in backtesting")

    async def get_balance(self) -> Dict[str, Any]:
        raise NotImplementedError("get_balance is not used in backtesting")

    async def get_current_price(
        self, 
        pair: str
    ) -> float:
        raise NotImplementedError("get_current_price is not used in backtesting")

    async def cancel_order(
        self, 
        order_id: str, 
        pair: str
    ) -> Dict[str, Union[str, float]]:
        raise NotImplementedError("cancel_order is not used in backtesting")

    async def get_exchange_status(self) -> dict:
        raise NotImplementedError("get_exchange_status is not used in backtesting")
    
    async def close_connection(self) -> None:
        self.logger.info("[BACKTEST] Closing WebSocket connection...")
        
    async def set_leverage(
        self,
        pair: str,
        leverage: int,
        margin_mode: str = 'isolated'
    ) -> Dict[str, Any]:
        """Sets the leverage and margin mode for a specific trading pair in backtesting."""
        self.logger.info(f"[BACKTEST] Setting leverage to {leverage}x and margin mode to {margin_mode} for {pair}")
        self.leverage = leverage
        self.margin_type = margin_mode
        
        # Store the leverage settings for this pair
        if pair not in self.positions:
            self.positions[pair] = {
                'leverage': leverage,
                'margin_mode': margin_mode,
                'long': None,
                'short': None
            }
        else:
            self.positions[pair]['leverage'] = leverage
            self.positions[pair]['margin_mode'] = margin_mode
            
        return {
            'leverage': leverage,
            'marginMode': margin_mode,
            'pair': pair,
            'success': True
        }
    
    async def get_positions(
        self,
        pair: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Fetches current open positions in backtesting, optionally filtered by trading pair."""
        self.logger.info(f"[BACKTEST] Getting positions for {pair if pair else 'all pairs'}")
        
        result = []
        
        # If a specific pair is requested
        if pair and pair in self.positions:
            for side in ['long', 'short']:
                position = self.positions[pair].get(side)
                if position and position.get('size', 0) > 0:
                    result.append(position)
        # If all positions are requested
        elif not pair:
            for p, position_data in self.positions.items():
                for side in ['long', 'short']:
                    position = position_data.get(side)
                    if position and position.get('size', 0) > 0:
                        result.append(position)
                        
        return result
    
    async def close_position(
        self,
        pair: str,
        position_side: Optional[str] = None
    ) -> Dict[str, Any]:
        """Closes an open position for the specified trading pair and side in backtesting."""
        self.logger.info(f"[BACKTEST] Closing position for {pair}, side: {position_side if position_side else 'all'}")
        
        if pair not in self.positions:
            return {"status": "no_position", "message": f"No open position found for {pair}"}
            
        sides_to_close = ['long', 'short'] if not position_side else [position_side.lower()]
        closed_positions = []
        
        for side in sides_to_close:
            if self.positions[pair].get(side) and self.positions[pair][side].get('size', 0) > 0:
                closed_position = self.positions[pair][side].copy()
                self.positions[pair][side] = None
                closed_positions.append(closed_position)
                
        if not closed_positions:
            return {"status": "no_position", "message": f"No {position_side if position_side else ''} position found for {pair}"}
            
        return {"status": "closed", "positions": closed_positions}
    
    async def get_funding_rate(
        self,
        pair: str
    ) -> Dict[str, Any]:
        """Fetches the current funding rate for a perpetual futures contract in backtesting."""
        self.logger.info(f"[BACKTEST] Getting funding rate for {pair}")
        
        # In backtesting, we can simulate a funding rate or use a fixed value
        return {
            'pair': pair,
            'fundingRate': 0.0001,  # Simulated funding rate (0.01% per 8 hours)
            'fundingTimestamp': int(time.time() * 1000),
            'nextFundingTimestamp': int(time.time() * 1000) + (8 * 60 * 60 * 1000),  # 8 hours later
            'previousFundingTimestamp': int(time.time() * 1000) - (8 * 60 * 60 * 1000)  # 8 hours ago
        }
    
    async def get_contract_specifications(
        self,
        pair: str
    ) -> Dict[str, Any]:
        """Fetches contract specifications for a futures contract in backtesting."""
        self.logger.info(f"[BACKTEST] Getting contract specifications for {pair}")
        
        # Return simulated contract specifications based on the pair
        base_currency, quote_with_settlement = pair.split('/')
        
        # Handle futures pairs format like "BTC/USDT:USDT"
        if ':' in quote_with_settlement:
            quote_currency, settlement_currency = quote_with_settlement.split(':')
        else:
            quote_currency = quote_with_settlement
            settlement_currency = quote_with_settlement
        
        return {
            "pair": pair,
            "contract_size": self.contract_size,
            "price_precision": 2,
            "amount_precision": 3,
            "minimum_amount": 0.001,
            "maximum_amount": 1000,
            "minimum_cost": 5,
            "maximum_leverage": 100,
            "maintenance_margin_rate": 0.005,  # 0.5%
            "is_inverse": False,
            "is_linear": True,
            "settlement_currency": settlement_currency
        }