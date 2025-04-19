from datamodel import OrderDepth, UserId, TradingState, Order, ConversionObservation
from typing import List, Dict, Any
import string
import jsonpickle
import json
import numpy as np
import math
import pandas as pd
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

BASKET1 = "PICNIC_BASKET1"
BASKET2 = "PICNIC_BASKET2"
CROISSANTS = "CROISSANTS"
DJEMBES = "DJEMBES"
JAMS = "JAMS"

WINDOW = 200
VOLUME_BASKET = 6

PRODUCTS = [
    CROISSANTS,
    JAMS,
    DJEMBES,
    BASKET1]

POSITION_LIMITS = {
    BASKET1: 60,
    BASKET2: 100,
    JAMS : 350,
    CROISSANTS : 250,
    DJEMBES : 60
}

class Trader:

    def __init__(self) -> None:
        
        print("Initializing Trader... ok")

        self.position_limit = {
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

        self.ema_param = 0.5

        self.prices = {
            "Spread":pd.Series(),
            "SPREAD_BASKET": pd.Series(),
        }

        self.all_positions = set()

        self.coconuts_pair_position = 0
        self.last_dolphin_price = -1
        self.dolphin_signal = 0 # 0 if closed, 1 long, -1 short
        self.trend = 0

        self.min_time_hold_position = 20 * 100
        self.initial_time_hold_position = 0
    def get_position(self, product, state : TradingState):
        return state.position.get(product, 0)  
    def get_mid_price(self, product, state : TradingState):

        market_bids = state.order_depths[product].buy_orders
        market_asks = state.order_depths[product].sell_orders
        best_bid = max(market_bids)
        best_ask = min(market_asks)
        return (best_bid + best_ask)/2
             
    def save_prices_product(
            self, 
            product, 
            state: TradingState,
            price):
            if not price:
                price = self.get_mid_price(product, state)

            self.prices[product] = pd.concat([
                self.prices[product],
                pd.Series({state.timestamp: price})
            ])

    def picnic_strategy(self, state: TradingState)-> List[Order]:
            orders_croissants = []
            orders_djembes = []
            orders_jams = []
            orders_basket = []
            
            def create_orders(buy_basket: bool)-> List[List[Order]]:

                if buy_basket:
                    sign = 1
                    price_basket = int (1e7)
                    price_others = 1
                else:
                    sign = -1
                    price_basket = 1
                    price_others = int (1e7)
                
                orders_basket.append(
                    Order(BASKET1, price_basket, sign*VOLUME_BASKET)
                )
                orders_croissants.append(
                    Order(CROISSANTS, price_others, -sign*6*VOLUME_BASKET)
                )
                orders_jams.append(
                    Order(JAMS, price_others, -sign*3*VOLUME_BASKET)
                )
                orders_djembes.append(
                    Order(DJEMBES, price_others, -sign*1*VOLUME_BASKET)
                )

                # return orders_basket, orders_croissants, orders_djembes, orders_jams
            

            price_basket = self.get_mid_price(BASKET1, state)
            price_croissants = self.get_mid_price(CROISSANTS, state)
            price_djembes = self.get_mid_price(DJEMBES, state)
            price_jams = self.get_mid_price(JAMS, state)

            position_basket = self.get_position(BASKET1, state)

            spread = price_basket - (6*price_croissants + 1* price_djembes + 3*price_jams)
            self.save_prices_product(
                "SPREAD_BASKET",
                state,
                spread
            )

            avg_spread = self.prices["SPREAD_BASKET"].rolling(WINDOW).mean()
            std_spread = self.prices["SPREAD_BASKET"].rolling(WINDOW).std()
            spread_5 = self.prices["SPREAD_BASKET"].rolling(45).mean()

            if not np.isnan(avg_spread.iloc[-1]):
                avg_spread = avg_spread.iloc[-1]
                std_spread = std_spread.iloc[-1]
                spread_5 = spread_5.iloc[-1]
                print(f"Average spread: {avg_spread}, Spread5: {spread_5}, Std: {std_spread}")

                threshold = 2.25
                if abs(position_basket) <= POSITION_LIMITS[BASKET1]-2:
                    if spread_5 < avg_spread - threshold*std_spread:  # buy basket
                        buy_basket = True
                        create_orders(buy_basket)

                    elif spread_5 > avg_spread + threshold*std_spread: # sell basket
                        buy_basket = False 
                        create_orders(buy_basket)

                else: # abs(position_basket) >= POSITION_LIMITS[PICNIC_BASKET]-10
                    if position_basket >0 : # sell basket
                        if spread_5 > avg_spread + threshold*std_spread:
                            buy_basket = False
                            create_orders(buy_basket)

                    else: # buy basket
                        if spread_5 < avg_spread - threshold*std_spread:
                            buy_basket = True
                            create_orders(buy_basket)
            return orders_croissants, orders_djembes, orders_jams, orders_basket

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        # Initialize the method output dict as an empty dict
        result = {}

        # PICNIC BASKET STRATEGY
        try:
            result[CROISSANTS], \
            result[DJEMBES], \
            result[JAMS], \
            result[BASKET1] = self.picnic_strategy(state)
        
        except Exception as e:
            print(e)


        logger.flush(state, result, conversions=0, trader_data="SAMPLE")
        return result, 0, "SAMPLE"
