from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict, Any
import string
import jsonpickle
import numpy as np
import math
import json

import pandas as pd

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, ConversionObservation


# --- Logger Class (Keep as provided, only logging format is fixed) ---
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
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

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
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

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
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

    def compress_observations(self, observations: Observation) -> list[Any]:
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

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()

SUBMISSION = "SUBMISSION"
class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP" 
    SQUID_INK = "SQUID_INK"
    CROISSANTS = "CROISSANTS"
    DJEMBES = "DJEMBES"
    JAMS = "JAMS"
    BASKET1 = "PICNIC_BASKET1"
    BASKET2 = "PICNIC_BASKET2"
    SPREAD = "SPREAD"
    SPREAD2 = "SPREAD2"
    ema_param = 0.33
    STRAWBERRIES = "STRAWBERRIES"
    ROSES = "ROSES"
    SYNTHETIC = "SYNTHETIC"
    SYNTHETIC2 = "SYNTHETIC2"
    SPREAD = "SPREAD"
    SPREAD2 = "SPREAD2"
    VOLCANIC_ROCK = "VOLCANIC_ROCK"
    MAGNIFICENT_MACARONS = "MAGNIFICENT_MACARONS"
    ORCHIDS = "MAGNIFICENT_MACARONS"
    
    
   

    ema_prices = {
        RAINFOREST_RESIN: None,
        KELP: None,
        SQUID_INK: None,
        CROISSANTS: None,
        DJEMBES :None,
        BASKET1: None,
        BASKET2:None,
        JAMS : None
        
    }



    past_prices = {
        RAINFOREST_RESIN: [],
        KELP: [],
        SQUID_INK: [],
        CROISSANTS: [],
        DJEMBES :[],
        BASKET1: [],
        BASKET2:[],
        JAMS : [],
        VOLCANIC_ROCK : []

    }

    vwap_history = {
        KELP : [],
        SQUID_INK : []
    }

    prev_mid_price = {
        JAMS : []
    }
PARAMS = {
    Product.ORCHIDS:{
        "make_edge": 8,
        "make_min_edge": 4,
        "make_probability": 0.566,
        "init_make_edge": 2,
        "min_edge": 0.5,
        "volume_avg_timestamp": 5,
        "volume_bar": 75,
        "dec_edge_discount": 0.8,
        "step_size":0.5
    }
}

class Trader:

    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params
        self.LIMIT = {Product.RAINFOREST_RESIN: 50 , 
                      Product.KELP : 50 , 
                      Product.SQUID_INK : 50 , 
                      Product.JAMS : 350 , 
                      Product.DJEMBES : 60,
                        Product.BASKET1: 60,
                        Product.BASKET2: 100,

                        Product.MAGNIFICENT_MACARONS: 75,
                        Product.STRAWBERRIES: 350,
                        Product.ROSES: 60,
                        Product.ORCHIDS : 75
    
                
                    }  # Max position allowed
        self.round = 0
        self.cash = 0
        self.indicator_cache = {}
        self.prices = {
            Product.JAMS: pd.Series(dtype=float),
            Product.CROISSANTS: pd.Series(dtype=float),
            Product.DJEMBES : pd.Series(dtype=float),
            "Spread": pd.Series(dtype=float),
            "Spread2": pd.Series(dtype=float)
        }

        self.Spread = {
            "spread_history": [],
                "prev_zscore": 0,
                "clear_flag": False,
                "curr_avg": 0,
                "curr_len" : 0,
                "inc_mean" : 0 
        }
        self.Spread2 = {
            "spread_history": [],
                "prev_zscore": 0,
                "clear_flag": False,
                "curr_avg": 0,
                "curr_len" : 0,
                "inc_mean" : 0 
        }

        # --- JAMS Strategy Parameters ---
        # Fixed range levels (can be adjusted)
        self.JAMS_FIXED_RANGE_BOTTOM = 6520
        self.JAMS_FIXED_RANGE_TOP = 6690

        # Parameters for Dynamic Trend Entry
        self.JAMS_SMA_WINDOW = 20       # Lookback period for Simple Moving Average
        self.JAMS_TREND_ENTRY_OFFSET = 3 # How many points below SMA to set the entry trigger

        # Parameters for Trailing Stop & Profit Taking
        self.JAMS_TRAILING_STOP_DISTANCE = 30
        self.JAMS_PARTIAL_PROFIT_TARGET = 6510
        self.JAMS_PARTIAL_PROFIT_FRACTION = 0.5

        # --- State Variables ---
        self.round = 0
        self.cash = 0 # Note: PnL calculation needs fixing if cash isn't updated properly
        self.previous_jams_position = 0
        self.jams_lowest_low_since_short = float('inf')
        self.jams_partial_profit_taken_this_move = False
        self.kelp_spread_margin = 1.5

       

        self.past_prices = {}
        self.recent_prices = {}
        if not hasattr(Product, 'past_prices'):
             Product.past_prices = {Product.DJEMBES: []}


        self.strike_prices = {
            "VOLCANIC_ROCK_VOUCHER_9500": 9500,
            "VOLCANIC_ROCK_VOUCHER_9750": 9750,
            "VOLCANIC_ROCK_VOUCHER_10000": 10000,
            "VOLCANIC_ROCK_VOUCHER_10250": 10250,
            "VOLCANIC_ROCK_VOUCHER_10500": 10500,
        }
        self.position_limits = {
            "VOLCANIC_ROCK": 400,
            "VOLCANIC_ROCK_VOUCHER_9500": 200,
            "VOLCANIC_ROCK_VOUCHER_9750": 200,
            "VOLCANIC_ROCK_VOUCHER_10000": 200,
            "VOLCANIC_ROCK_VOUCHER_10250": 200,
            "VOLCANIC_ROCK_VOUCHER_10500": 200,
        }
        self.price_history = []
        self.risk_free_rate = 0.01
        self.days_total = 7
        self.days_to_expiry = self.days_total


        self.profit_target = 5 # Units of price
        self.stop_loss = 5  # Units of price
        self.price_min = 6495
        self.price_max = 6555
        self.position = {Product.JAMS: 0}
        self.entry_price = {Product.JAMS: None}
        self.historical_data_length = 20
        self.order_size = 175 # Default order size
        self.k_factor = 0.007  # Sensitivity factor fo
        self.traderObject = {}


    def orchids_implied_bid_ask(
        self,
        observation: ConversionObservation,
    ) -> (float, float):
        return observation.bidPrice - observation.exportTariff - observation.transportFees - 0.1, observation.askPrice + observation.importTariff + observation.transportFees

    def orchids_adap_edge(
        self,
        timestamp: int,
        curr_edge: float,
        position: int,
        traderObject: dict
    ) -> float: 
        if timestamp == 0:
            traderObject["ORCHIDS"]["curr_edge"] = self.params[Product.ORCHIDS]["init_make_edge"]
            return self.params[Product.ORCHIDS]["init_make_edge"]

        # Timestamp not 0
        traderObject["ORCHIDS"]["volume_history"].append(abs(position))
        if len(traderObject["ORCHIDS"]["volume_history"]) > self.params[Product.ORCHIDS]["volume_avg_timestamp"]:
            traderObject["ORCHIDS"]["volume_history"].pop(0)

        if len(traderObject["ORCHIDS"]["volume_history"]) < self.params[Product.ORCHIDS]["volume_avg_timestamp"]:
            return curr_edge
        elif not traderObject["ORCHIDS"]["optimized"]:
            volume_avg = np.mean(traderObject["ORCHIDS"]["volume_history"])

            # Bump up edge if consistently getting lifted full size
            if volume_avg >= self.params[Product.ORCHIDS]["volume_bar"]:
                traderObject["ORCHIDS"]["volume_history"] = [] # clear volume history if edge changed
                traderObject["ORCHIDS"]["curr_edge"] = curr_edge + self.params[Product.ORCHIDS]["step_size"]
                return curr_edge + self.params[Product.ORCHIDS]["step_size"]

            # Decrement edge if more cash with less edge, included discount
            elif self.params[Product.ORCHIDS]["dec_edge_discount"] * self.params[Product.ORCHIDS]["volume_bar"] * (curr_edge - self.params[Product.ORCHIDS]["step_size"]) > volume_avg * curr_edge:
                if curr_edge - self.params[Product.ORCHIDS]["step_size"] > self.params[Product.ORCHIDS]["min_edge"]:
                    traderObject["ORCHIDS"]["volume_history"] = [] # clear volume history if edge changed
                    traderObject["ORCHIDS"]["curr_edge"] = curr_edge - self.params[Product.ORCHIDS]["step_size"]
                    traderObject["ORCHIDS"]["optimized"] = True
                    return curr_edge - self.params[Product.ORCHIDS]["step_size"]
                else:
                    traderObject["ORCHIDS"]["curr_edge"] = self.params[Product.ORCHIDS]["min_edge"]
                    return self.params[Product.ORCHIDS]["min_edge"]

        traderObject["ORCHIDS"]["curr_edge"] = curr_edge
        return curr_edge

    def orchids_arb_take(
        self,
        order_depth: OrderDepth,
        observation: ConversionObservation,
        adap_edge: float,
        position: int
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        position_limit = self.LIMIT[Product.ORCHIDS]
        buy_order_volume = 0
        sell_order_volume = 0

        implied_bid, implied_ask = self.orchids_implied_bid_ask(observation)                                                                                                                                                                    

        buy_quantity = position_limit - position
        sell_quantity = position_limit + position

        ask = implied_ask + adap_edge

        foreign_mid = (observation.askPrice + observation.bidPrice) / 2
        aggressive_ask = foreign_mid - 1.6

        if aggressive_ask > implied_ask:
            ask = aggressive_ask

        edge = (ask - implied_ask) * self.params[Product.ORCHIDS]["make_probability"]

        for price in sorted(list(order_depth.sell_orders.keys())):
            if price > implied_bid - edge:
                break

            if price < implied_bid - edge:
                quantity = min(abs(order_depth.sell_orders[price]), buy_quantity) # max amount to buy
                if quantity > 0:
                    orders.append(Order(Product.ORCHIDS, round(price), quantity))
                    buy_order_volume += quantity

        for price in sorted(list(order_depth.buy_orders.keys()), reverse=True):
            if price < implied_ask + edge:
                break

            if price > implied_ask + edge:
                quantity = min(abs(order_depth.buy_orders[price]), sell_quantity) # max amount to sell
                if quantity > 0:
                    orders.append(Order(Product.ORCHIDS, round(price), -quantity))
                    sell_order_volume += quantity

        return orders, buy_order_volume, sell_order_volume

    def orchids_arb_clear(
        self,
        position: int
    ) -> int:
        conversions = -position
        return conversions

    def orchids_arb_make(
        
        self,
        order_depth: OrderDepth,
        observation: ConversionObservation,
        position: int,
        edge: float,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        position_limit = self.LIMIT[Product.ORCHIDS]

        # Implied Bid = observation.bidPrice - observation.exportTariff - observation.transportFees - 0.1
        # Implied Ask = observation.askPrice + observation.importTariff + observation.transportFees
        implied_bid, implied_ask = self.orchids_implied_bid_ask(observation)

        bid = implied_bid - edge
        ask = implied_ask + edge

        # ask = foreign_mid - 1.6 best performance so far
        foreign_mid = (observation.askPrice + observation.bidPrice) / 2
        aggressive_ask = foreign_mid - 1.6 # Aggressive ask

        # don't lose money
        if aggressive_ask >= implied_ask + self.params[Product.ORCHIDS]['min_edge']:
            ask = aggressive_ask
            print("AGGRESSIVE")
            print(f"ALGO ASK: {round(ask)}")
            print(f"ALGO BID: {round(bid)}")
        else:
            print(f"ALGO ASK: {round(ask)}")
            print(f"ALGO BID: {round(bid)}")

        filtered_ask = [price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= 40]
        filtered_bid = [price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= 25]

        # If we're not best level, penny until min edge
        if len(filtered_ask) > 0 and ask > filtered_ask[0]:
            if filtered_ask[0] - 1 > implied_ask:
                ask = filtered_ask[0] - 1
            else:
                ask = implied_ask + edge
        if len(filtered_bid) > 0 and  bid < filtered_bid[0]:
            if filtered_bid[0] + 1 < implied_bid:
                bid = filtered_bid[0] + 1
            else:
                bid = implied_bid - edge

        print(f"IMPLIED_BID: {implied_bid}")
        print(f"IMPLIED_ASK: {implied_ask}")
        print(f"FOREIGN ASK: {observation.askPrice}")
        print(f"FOREIGN BID: {observation.bidPrice}")

        best_bid = min(order_depth.buy_orders.keys())
        best_ask = max(order_depth.sell_orders.keys())

        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(Product.ORCHIDS, round(bid), buy_quantity))  # Buy order

        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(Product.ORCHIDS, round(ask), -sell_quantity))  # Sell order

        return orders, buy_order_volume, sell_order_volume
    

    def run(self,state): 
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        result = {}
        conversions = 0

        if Product.ORCHIDS in self.params and Product.ORCHIDS in state.order_depths:
            if "ORCHIDS" not in traderObject:
                traderObject["ORCHIDS"] = {"curr_edge": self.params[Product.ORCHIDS]["init_make_edge"], "volume_history": [], "optimized": False}
            orchids_position = (
                state.position[Product.ORCHIDS]
                if Product.ORCHIDS in state.position
                else 0
            )
            print(f"ORCHIDS POSITION: {orchids_position}")

            conversions = self.orchids_arb_clear(
                orchids_position
            )

            adap_edge = self.orchids_adap_edge(
                state.timestamp,
                traderObject["ORCHIDS"]["curr_edge"],
                orchids_position,
                traderObject,
            )

            orchids_position = 0

            orchids_take_orders, buy_order_volume, sell_order_volume = self.orchids_arb_take(
                state.order_depths[Product.ORCHIDS],
                state.observations.conversionObservations[Product.ORCHIDS],
                adap_edge,
                orchids_position,
            )

            orchids_make_orders, _, _ = self.orchids_arb_make(
                state.order_depths[Product.ORCHIDS],
                state.observations.conversionObservations[Product.ORCHIDS],
                orchids_position,
                adap_edge,
                buy_order_volume,
                sell_order_volume
            )

            result[Product.ORCHIDS] = (
                orchids_take_orders + orchids_make_orders
            )

        traderData = jsonpickle.encode(traderObject)

        return result, conversions, traderData