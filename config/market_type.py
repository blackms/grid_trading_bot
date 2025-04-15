from enum import Enum

class MarketType(Enum):
    SPOT = "spot"
    FUTURES = "futures"

    @staticmethod
    def from_string(market_type_str: str):
        try:
            return MarketType(market_type_str)
        except ValueError:
            raise ValueError(f"Invalid market type: '{market_type_str}'. Available market types are: {', '.join([market_type.value for market_type in MarketType])}")