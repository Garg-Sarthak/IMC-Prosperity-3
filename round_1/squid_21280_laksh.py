from typing import Dict, List
import numpy as np
import json
import math
from datamodel import OrderDepth, TradingState, Order, Trade
# from scipy.stats import linregress
from typing import Any
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders, conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings) -> list[list[any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths):
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades) -> list[list[any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders) -> list[list[any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()


SUBMISSION = "SUBMISSION"
KELP = "KELP"
RESIN = "RAINFOREST_RESIN"
SQUID = "SQUID_INK"
PRODUCTS = [
    KELP,
    RESIN,
    SQUID
]

DEFAULT_PRICES = {
    RESIN : 10000,
    KELP : 2016,
    SQUID : 2040
}


class Trader:

    def __init__(self) -> None:
        
        logger.print("Initializing Trader...")

        self.position_limit = {
            RESIN : 50,
            KELP : 50,
            SQUID : 50
        }

        
        # self.past_prices keeps the list of all past prices
        self.past_prices = dict()
        for product in PRODUCTS:
            self.past_prices[product] = []

        # self.ema_prices keeps an exponential moving average of prices
        self.ema_prices = dict()
        for product in PRODUCTS:
            self.ema_prices[product] = None

    

        self.trend_sma_period = 50       # Lookback period for the trend SMA
        self.trend_ = 1.3 # Std dev multiplier for MR entries
        self.mr_target_position_size = 35 # Target size for MR entries (<= limit)
        self.past_long_sma = dict()
        for product in PRODUCTS:
            self.past_long_sma[product] = []

  


    def get_position(self, product, state : TradingState):
        return state.position.get(product, 0)    

    def get_mid_price(self, product, state : TradingState):

        default_price = self.ema_prices[product]
        if default_price is None:
            default_price = DEFAULT_PRICES[product]

        if product not in state.order_depths:
            return default_price

        market_bids = state.order_depths[product].buy_orders
        if len(market_bids) == 0:
            # There are no bid orders in the market (midprice undefined)
            return default_price
        
        market_asks = state.order_depths[product].sell_orders
        if len(market_asks) == 0:
            # There are no bid orders in the market (mid_price undefined)
            return default_price
        
        best_bid = max(market_bids)
        best_ask = min(market_asks)
        return (best_bid + best_ask)/2



    def update_past_prices(self,state:TradingState) -> None :
        for product in PRODUCTS:
            mid_price = self.get_mid_price(product,state)
            if mid_price is not None :
                self.past_prices[product].append(mid_price)
        return
    
    def calculate_sma(self, prices: List[float], period: int):

        if period <= 0:
             logger.print(f"Error: SMA period must be positive (got {period}).")
             return None
        if len(prices) < period:
            # Not enough historical data to calculate the SMA for the given period
            return None
        else:
            # Select the most recent 'period' prices and calculate their mean
            sma_value = np.mean(prices[-period:])
            return sma_value

    
    def get_trend_slope(prices, window=10):
            if len(prices) < window:
                return 0
            x = list(range(window))
            y = prices[-window:]
            slope, _, r_value, _, _ = linregress(x, y)
            return slope * r_value  # slope * correlation gives strength + direction
    


    def strat2(self, state: TradingState, product):
        position = state.position.get(product, 0)
        orders = []

        buy_qty = self.position_limit[product] - position
        sell_qty = -self.position_limit[product] - position

        market_bids = state.order_depths[product].buy_orders.keys()
        market_asks = state.order_depths[product].sell_orders.keys()

        mid_price = self.get_mid_price(product, state)

        best_bid = max(market_bids)
        best_ask = min(market_asks)

        residuals = [p for p in self.past_prices[product][-50:]]
        recent_prices = self.past_prices[product][-10:]

        std = np.std(residuals)
        mn = np.mean(residuals)
        slope = recent_prices[-1] - recent_prices[0]    # delta of prices
        threshold = std * 3.2   # random value - to tune

        logger.print(f"mean : {mn}, std : {std}, slope: {slope}, threshold : {threshold}")

        # === Trendy Phase ===

        #going up
        if slope > threshold:
            if position >= 0:
                # Exit longs if you're already holding
                orders.append(Order(product, best_ask, -position))
                logger.print(f"Market trending UP - Sell holdings at {best_ask}")
            else:
                # Mini momentum scalp - trend up
                scalp_qty = min(20,50 - position )
                if scalp_qty > 0:
                    orders.append(Order(product, best_bid + 1, scalp_qty))
                    logger.print(f"Scalp BUY {scalp_qty} at {best_bid + 1} on UP momentum")

        elif slope < -threshold:
            if position <= 0:
                # Exit shorts
                orders.append(Order(product, best_bid,  position))
                logger.print(f"Market trending DOWN - Buy back at {best_bid}")
            else:
                # Mini momentum scalp - trend down
                scalp_qty = min(-50-position,-20)
                if scalp_qty > 0:
                    orders.append(Order(product, best_ask - 1, scalp_qty))
                    logger.print(f"Scalp SELL {scalp_qty} at {best_ask - 1} on DOWN momentum")

        # === Ranging Market ===
        else:
            if mid_price >= mn + std*1.45 :
                orders.append(Order(product, best_ask, sell_qty))
                logger.print(f"sell at {best_bid} : {sell_qty}")
            elif mid_price <= mn - std*1.45:
                orders.append(Order(product, best_bid, buy_qty))
                logger.print(f"buy at {best_ask} : {buy_qty}")

        return orders


    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        # self.round += 1


        self.update_past_prices(state)

        # Initialize the method output dict as an empty dict
        result = {}

        # # KELP STRATEGY
        # try:
        #     result[KELP] = self.squid_s(state,KELP)
        # except Exception as e:
        #     logger.print("Error in KELP strategy")
        #     logger.print(e)

        try:
            result[SQUID] = self.strat2(state,SQUID)
        except Exception as e:
            logger.print("Error in squid strategy")
            logger.print(e)

        logger.print("+---------------------------------+")
        logger.flush(state, result, conversions=0, trader_data="SAMPLE")

        return result, 0, "SAMPLE"