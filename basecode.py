from typing import Dict, List
import numpy as np
import json
import math
from datamodel import OrderDepth, TradingState, Order, Trade

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
PRODUCT2 = "KELP" 
PRODUCT1 = "RAINFOREST_RESIN"

PRODUCTS = [
    PRODUCT1,
    PRODUCT2,
]

DEFAULT_PRICES = {
    PRODUCT1 : 10_000,
    PRODUCT2 : 2_020,
}

class Trader:

    def __init__(self) -> None:
        
        logger.print("Initializing Trader...")

        self.position_limit = {
            PRODUCT1 : 20,
            PRODUCT2 : 20,
        }

        self.round = 0

        # Values to compute pnl
        self.cash = 0
        # positions can be obtained from state.position
        
        # self.past_prices keeps the list of all past prices
        self.past_prices = dict()
        for product in PRODUCTS:
            self.past_prices[product] = []

        # self.ema_prices keeps an exponential moving average of prices
        self.ema_prices = dict()
        for product in PRODUCTS:
            self.ema_prices[product] = None

        self.ema_param = 0.39


    # utils
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

    def get_value_on_product(self, product, state : TradingState):
        """
        Returns the amount of MONEY currently held on the product.  
        """
        return self.get_position(product, state) * self.get_mid_price(product, state)
            
    def update_pnl(self, state : TradingState):
        """
        Updates the pnl.
        """
        def update_cash():
            # Update cash
            for product in state.own_trades:
                for trade in state.own_trades[product]:
                    if trade.timestamp != state.timestamp - 100:
                        # Trade was already analyzed
                        continue

                    if trade.buyer == SUBMISSION:
                        self.cash -= trade.quantity * trade.price
                    if trade.seller == SUBMISSION:
                        self.cash += trade.quantity * trade.price
        
        def get_value_on_positions():
            value = 0
            for product in state.position:
                value += self.get_value_on_product(product, state)
            return value
        
        # Update cash
        update_cash()
        return self.cash + get_value_on_positions()

    def update_ema_prices(self, state : TradingState):
        """
        Update the exponential moving average of the prices of each product.
        """
        for product in PRODUCTS:
            mid_price = self.get_mid_price(product, state)
            if mid_price is None:
                continue

            # Update ema price
            if self.ema_prices[product] is None:
                self.ema_prices[product] = mid_price
            else:
                self.ema_prices[product] = self.ema_param * mid_price + (1-self.ema_param) * self.ema_prices[product]

    def pearls_strategy(self, state : TradingState):
        """
        Returns a list of orders with trades of PRODUCT1.

        Comment: Mudar depois. Separar estrategia por produto assume que
        cada produto eh tradado independentemente
        """

        position_pearls = self.get_position(PRODUCT1, state)

        bid_volume = self.position_limit[PRODUCT1] - position_pearls
        ask_volume = - self.position_limit[PRODUCT1] - position_pearls

        orders = []
        orders.append(Order(PRODUCT1, DEFAULT_PRICES[PRODUCT1] - 4, bid_volume))
        orders.append(Order(PRODUCT1, DEFAULT_PRICES[PRODUCT1] + 4, ask_volume))

        return orders

    def bananas_strategy(self, state : TradingState):
        """
        Returns a list of orders with trades of PRODUCT2.

        Comment: Mudar depois. Separar estrategia por produto assume que
        cada produto eh tradado independentemente
        """

        position_bananas = self.get_position(PRODUCT2, state)

        bid_volume = self.position_limit[PRODUCT2] - position_bananas
        ask_volume = - self.position_limit[PRODUCT2] - position_bananas

        orders = []

        if position_bananas == 0:
            # Not long nor short
            orders.append(Order(PRODUCT2, math.floor(self.ema_prices[PRODUCT2] - 1), bid_volume))
            orders.append(Order(PRODUCT2, math.ceil(self.ema_prices[PRODUCT2] + 1), ask_volume))
        
        if position_bananas > 0:
            # Long position
            orders.append(Order(PRODUCT2, math.floor(self.ema_prices[PRODUCT2] - 1), bid_volume))
            orders.append(Order(PRODUCT2, math.ceil(self.ema_prices[PRODUCT2]), ask_volume))

        if position_bananas < 0:
            # Short position
            orders.append(Order(PRODUCT2, math.floor(self.ema_prices[PRODUCT2]), bid_volume))
            orders.append(Order(PRODUCT2, math.ceil(self.ema_prices[PRODUCT2] + 1), ask_volume))

        return orders


    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """

        self.round += 1
        pnl = self.update_pnl(state)
        self.update_ema_prices(state)

        logger.print(f"Log round {self.round}")

        logger.print("TRADES:")
        for product in state.own_trades:
            for trade in state.own_trades[product]:
                if trade.timestamp == state.timestamp - 100:
                    logger.print(trade)

        logger.print(f"\tCash {self.cash}")
        for product in PRODUCTS:
            logger.print(f"\tProduct {product}, Position {self.get_position(product, state)}, Midprice {self.get_mid_price(product, state)}, Value {self.get_value_on_product(product, state)}, EMA {self.ema_prices[product]}")
        logger.print(f"\tPnL {pnl}")
        

        # Initialize the method output dict as an empty dict
        result = {}

        # PEARL STRATEGY
        try:
            result[PRODUCT1] = self.pearls_strategy(state)
        except Exception as e:
            logger.print("Error in PRODUCT1 strategy")
            logger.print(e)

        # BANANA STRATEGY
        try:
            result[PRODUCT2] = self.bananas_strategy(state)
        except Exception as e:
            logger.print("Error in PRODUCT2 strategy")
            logger.print(e)

        logger.print("+---------------------------------+")
        logger.flush(state, result, conversions=0, trader_data="SAMPLE")

        return result, 0, "SAMPLE"