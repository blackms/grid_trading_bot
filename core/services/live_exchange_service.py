import ccxt, logging, asyncio, os
from ccxt.base.errors import NetworkError, BaseError, ExchangeError, OrderNotFound
import ccxt.pro as ccxtpro
from typing import Dict, Union, Callable, Any, Optional, List
import pandas as pd
from config.config_manager import ConfigManager
from config.market_type import MarketType
from .exchange_interface import ExchangeInterface
from .exceptions import UnsupportedExchangeError, DataFetchError, OrderCancellationError, MissingEnvironmentVariableError

class LiveExchangeService(ExchangeInterface):
    def __init__(
        self,
        config_manager: ConfigManager,
        is_paper_trading_activated: bool
    ):
        self.config_manager = config_manager
        self.is_paper_trading_activated = is_paper_trading_activated
        self.logger = logging.getLogger(self.__class__.__name__)
        self.exchange_name = self.config_manager.get_exchange_name()
        self.api_key = self._get_env_variable("EXCHANGE_API_KEY")
        self.secret_key = self._get_env_variable("EXCHANGE_SECRET_KEY")
        self.market_type = self.config_manager.get_market_type()
        self.exchange = self._initialize_exchange()
        self.connection_active = False
        
        # Initialize futures settings if applicable
        if self.market_type == MarketType.FUTURES:
            self._initialize_futures_settings()
    
    def _get_env_variable(self, key: str) -> str:
        value = os.getenv(key)
        if value is None:
            raise MissingEnvironmentVariableError(f"Missing required environment variable: {key}")
        return value

    def _initialize_exchange(self) -> None:
        try:
            options = {
                'enableRateLimit': True
            }
            
            # Add futures-specific options if needed
            if self.market_type == MarketType.FUTURES:
                options['options'] = {
                    'defaultType': 'future',
                    'marginType': self.config_manager.get_margin_type(),
                    'hedgeMode': self.config_manager.is_hedge_mode_enabled()
                }
            
            exchange = getattr(ccxtpro, self.exchange_name)({
                'apiKey': self.api_key,
                'secret': self.secret_key,
                **options
            })

            if self.is_paper_trading_activated:
                self._enable_sandbox_mode(exchange)
                
            return exchange
        except AttributeError:
            raise UnsupportedExchangeError(f"The exchange '{self.exchange_name}' is not supported.")
            
    def _initialize_futures_settings(self) -> None:
        """Initialize futures-specific settings after exchange initialization."""
        self.logger.info("Initializing futures trading settings")
        # This will be called asynchronously in the first operation
        self._futures_initialized = False

    def _enable_sandbox_mode(self, exchange) -> None:
        if self.exchange_name == 'binance':
            exchange.urls['api'] = 'https://testnet.binance.vision/api'
        elif self.exchange_name == 'kraken':
            exchange.urls['api'] = 'https://api.demo-futures.kraken.com'
        elif self.exchange_name == 'bitmex':
            exchange.urls['api'] = 'https://testnet.bitmex.com'
        elif self.exchange_name == 'bybit':
            exchange.set_sandbox_mode(True)
        else:
            self.logger.warning(f"No sandbox mode available for {self.exchange_name}. Running in live mode.")
    
    async def _subscribe_to_ticker_updates(
        self,
        pair: str, 
        on_ticker_update: Callable[[float], None], 
        update_interval: float,
        max_retries: int = 5
    ) -> None:
        self.connection_active = True
        retry_count = 0
        
        while self.connection_active:
            try:
                ticker = await self.exchange.watch_ticker(pair)
                current_price: float = ticker['last']
                self.logger.info(f"Connected to WebSocket for {pair} ticker current price: {current_price}")

                if not self.connection_active:
                    break

                await on_ticker_update(current_price)
                await asyncio.sleep(update_interval)
                retry_count = 0  # Reset retry count after a successful operation

            except (NetworkError, ExchangeError) as e:
                retry_count += 1
                retry_interval = min(retry_count * 5, 60)
                self.logger.error(f"Error connecting to WebSocket for {pair}: {e}. Retrying in {retry_interval} seconds ({retry_count}/{max_retries}).")
                
                if retry_count >= max_retries:
                    self.logger.error("Max retries reached. Stopping WebSocket connection.")
                    self.connection_active = False
                    break

                await asyncio.sleep(retry_interval)
            
            except asyncio.CancelledError:
                self.logger.error(f"WebSocket subscription for {pair} was cancelled.")
                self.connection_active = False
                break

            except Exception as e:
                self.logger.error(f"WebSocket connection error: {e}. Reconnecting...")
                await asyncio.sleep(5)

            finally:
                if not self.connection_active:
                    try:
                        self.logger.info("Connection to Websocket no longer active.")
                        await self.exchange.close()

                    except Exception as e:
                        self.logger.error(f"Error while closing WebSocket connection: {e}", exc_info=True)

    async def listen_to_ticker_updates(
        self, 
        pair: str, 
        on_price_update: Callable[[float], None],
        update_interval: float
    ) -> None:
        await self._subscribe_to_ticker_updates(pair, on_price_update, update_interval)

    async def close_connection(self) -> None:
        self.connection_active = False
        self.logger.info("Closing WebSocket connection...")

    async def get_balance(self) -> Dict[str, Any]:
        try:
            balance = await self.exchange.fetch_balance()
            return balance

        except BaseError as e:
            raise DataFetchError(f"Error fetching balance: {str(e)}")
    
    async def get_current_price(self, pair: str) -> float:
        try:
            ticker = await self.exchange.fetch_ticker(pair)
            return ticker['last']

        except BaseError as e:
            raise DataFetchError(f"Error fetching current price: {str(e)}")

    async def place_order(
        self,
        pair: str,
        order_type: str,
        order_side: str,
        amount: float,
        price: Optional[float] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Union[str, float]]:
        try:
            # Initialize futures settings if needed and not already initialized
            if self.market_type == MarketType.FUTURES and not self._futures_initialized:
                await self._ensure_futures_initialized(pair)
                
            # Use default empty dict if params is None
            params = params or {}
            
            order = await self.exchange.create_order(pair, order_type, order_side, amount, price, params)
            return order

        except NetworkError as e:
            raise DataFetchError(f"Network issue occurred while placing order: {str(e)}")

        except BaseError as e:
            raise DataFetchError(f"Error placing order: {str(e)}")

        except Exception as e:
            raise DataFetchError(f"Unexpected error placing order: {str(e)}")

    async def fetch_order(
        self, 
        order_id: str,
        pair: str
    ) -> Dict[str, Union[str, float]]:
        try:
            return await self.exchange.fetch_order(order_id, pair)

        except NetworkError as e:
            raise DataFetchError(f"Network issue occurred while fetching order status: {str(e)}")

        except BaseError as e:
            raise DataFetchError(f"Exchange-specific error occurred: {str(e)}")

        except Exception as e:
            raise DataFetchError(f"Failed to fetch order status: {str(e)}")

    async def cancel_order(
        self, 
        order_id: str, 
        pair: str
    ) -> dict:
        try:
            self.logger.info(f"Attempting to cancel order {order_id} for pair {pair}")
            cancellation_result = await self.exchange.cancel_order(order_id, pair)
            
            if cancellation_result['status'] in ['canceled', 'closed']:
                self.logger.info(f"Order {order_id} successfully canceled.")
                return cancellation_result
            else:
                self.logger.warning(f"Order {order_id} cancellation status: {cancellation_result['status']}")
                return cancellation_result

        except OrderNotFound as e:
            raise OrderCancellationError(f"Order {order_id} not found for cancellation. It may already be completed or canceled.")

        except NetworkError as e:
            raise OrderCancellationError(f"Network error while canceling order {order_id}: {str(e)}")

        except BaseError as e:
            raise OrderCancellationError(f"Exchange error while canceling order {order_id}: {str(e)}")

        except Exception as e:
            raise OrderCancellationError(f"Unexpected error while canceling order {order_id}: {str(e)}")
    
    async def get_exchange_status(self) -> dict:
        try:
            status = await self.exchange.fetch_status()
            return {
                "status": status.get("status", "unknown"),
                "updated": status.get("updated"),
                "eta": status.get("eta"),
                "url": status.get("url"),
                "info": status.get("info", "No additional info available")
            }

        except AttributeError:
            return {"status": "unsupported", "info": "fetch_status not supported by this exchange."}

        except Exception as e:
            return {"status": "error", "info": f"Failed to fetch exchange status: {e}"}

    def fetch_ohlcv(
        self, 
        pair: str, 
        timeframe: str, 
        start_date: str, 
        end_date: str
    ) -> pd.DataFrame:
        raise NotImplementedError("fetch_ohlcv is not used in live or paper trading mode.")
        
    async def _ensure_futures_initialized(self, pair: str) -> None:
        """Ensure futures settings are initialized for the given pair."""
        if self._futures_initialized:
            return
            
        try:
            # Set leverage and margin mode
            await self.set_leverage(
                pair,
                self.config_manager.get_leverage(),
                self.config_manager.get_margin_type()
            )
            
            # Set hedge mode if needed
            if self.config_manager.is_hedge_mode_enabled():
                await self.exchange.set_position_mode(True)  # Enable hedge mode
                
            self._futures_initialized = True
            self.logger.info(f"Futures settings initialized for {pair}")
        except Exception as e:
            self.logger.error(f"Failed to initialize futures settings: {e}")
            raise
    
    async def set_leverage(
        self,
        pair: str,
        leverage: int,
        margin_mode: str = 'isolated'
    ) -> Dict[str, Any]:
        """Sets the leverage and margin mode for a specific trading pair."""
        try:
            # Set margin mode first (isolated or cross)
            await self.exchange.set_margin_mode(margin_mode, pair)
            self.logger.info(f"Set margin mode to {margin_mode} for {pair}")
            
            # Then set leverage
            result = await self.exchange.set_leverage(leverage, pair)
            self.logger.info(f"Set leverage to {leverage}x for {pair}")
            
            return result
        except BaseError as e:
            self.logger.error(f"Error setting leverage for {pair}: {e}")
            raise DataFetchError(f"Failed to set leverage: {str(e)}")
    
    async def get_positions(
        self,
        pair: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Fetches current open positions, optionally filtered by trading pair."""
        try:
            positions = await self.exchange.fetch_positions(pair)
            
            # Filter out positions with zero size
            active_positions = [
                position for position in positions
                if float(position.get('contracts', 0)) > 0 or float(position.get('size', 0)) > 0
            ]
            
            return active_positions
        except BaseError as e:
            raise DataFetchError(f"Error fetching positions: {str(e)}")
    
    async def close_position(
        self,
        pair: str,
        position_side: Optional[str] = None
    ) -> Dict[str, Any]:
        """Closes an open position for the specified trading pair and side."""
        try:
            # Get current position
            positions = await self.get_positions(pair)
            
            if not positions:
                self.logger.warning(f"No open position found for {pair}")
                return {"status": "no_position", "message": f"No open position found for {pair}"}
            
            # Filter by side if specified
            if position_side:
                positions = [p for p in positions if p.get('side', '').lower() == position_side.lower()]
                
                if not positions:
                    self.logger.warning(f"No {position_side} position found for {pair}")
                    return {"status": "no_position", "message": f"No {position_side} position found for {pair}"}
            
            results = []
            for position in positions:
                # Determine the order side to close the position
                side = 'sell' if position.get('side', '').lower() == 'long' else 'buy'
                amount = abs(float(position.get('contracts', 0)) or float(position.get('size', 0)))
                
                # Place a market order to close the position
                close_order = await self.place_order(
                    pair=pair,
                    order_type='market',
                    order_side=side,
                    amount=amount,
                    params={"reduceOnly": True}
                )
                
                results.append(close_order)
                
            return {"status": "closed", "orders": results}
            
        except BaseError as e:
            raise DataFetchError(f"Error closing position: {str(e)}")
    
    async def get_funding_rate(
        self,
        pair: str
    ) -> Dict[str, Any]:
        """Fetches the current funding rate for a perpetual futures contract."""
        try:
            funding_info = await self.exchange.fetch_funding_rate(pair)
            return funding_info
        except BaseError as e:
            raise DataFetchError(f"Error fetching funding rate: {str(e)}")
    
    async def get_contract_specifications(
        self,
        pair: str
    ) -> Dict[str, Any]:
        """Fetches contract specifications for a futures contract."""
        try:
            # Get market information which includes contract specifications
            market = await self.exchange.fetch_market(pair)
            
            # Extract relevant contract specifications
            specs = {
                "pair": pair,
                "contract_size": market.get('contractSize', 1),
                "price_precision": market.get('precision', {}).get('price', 0),
                "amount_precision": market.get('precision', {}).get('amount', 0),
                "minimum_amount": market.get('limits', {}).get('amount', {}).get('min', 0),
                "maximum_amount": market.get('limits', {}).get('amount', {}).get('max', 0),
                "minimum_cost": market.get('limits', {}).get('cost', {}).get('min', 0),
                "maximum_leverage": market.get('limits', {}).get('leverage', {}).get('max', 1),
                "maintenance_margin_rate": market.get('info', {}).get('maintMarginRate', 0),
                "is_inverse": market.get('inverse', False),
                "is_linear": market.get('linear', True),
                "settlement_currency": market.get('settleId', market.get('quote', ''))
            }
            
            return specs
        except BaseError as e:
            raise DataFetchError(f"Error fetching contract specifications: {str(e)}")