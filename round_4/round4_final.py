from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict, Any
import string
import jsonpickle
import numpy as np
import math
import json

import pandas as pd

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


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

def round_price(p): return int(round(p))

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


ORDER_VOLUME = 5
WINDOW = 150

POSITION_LIMITS = {
    Product.CROISSANTS: 250,
    Product.JAMS: 350,
    Product.DJEMBES : 60
}

 

BASKET_1_WEIGHTS = {
    Product.CROISSANTS : 6,
    Product.JAMS : 3,
    Product.DJEMBES : 1
}

BASKET_2_WEIGHTS = {
    Product.CROISSANTS : 4,
    Product.JAMS : 2,
}
PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,  # Example fair value, could be dynamic
        "take_width": 1,      # How far from fair value to take aggressive orders
        "clear_width": 2,     # How far from fair value to clear inventory aggressively
        # for making
        "disregard_edge": 1,  # Don't penny/join levels this close to fair value
        "join_edge": 5,       # Join levels within this edge DEFAULT was 2
        "default_edge": 4,    # Default distance from fair value to place orders
        "soft_position_limit": 50,  # Adjust price slightly if position exceeds this
        "volume_limit_factor": 15,  # Factor to multiply opposite volume by for our quantity limit
                                     # 1.0 means limit to exact opposite volume
                                     # > 1.0 allows placing more, < 1.0 less
                                     # Set to None or a very large number to disable volume limit
    },
    # KELP params removed for clarity if not needed
    Product.KELP: {
        "fair_value": 2045,  # Example fair value, could be dynamic
        "take_width": 1,      # How far from fair value to take aggressive orders
        "clear_width": 2,     # How far from fair value to clear inventory aggressively
        # for making
        "disregard_edge": 1,  # Don't penny/join levels this close to fair value
        "join_edge": 5,       # Join levels within this edge DEFAULT was 2
        "default_edge": 4,    # Default distance from fair value to place orders
        "soft_position_limit": 50,  # Adjust price slightly if position exceeds this
        "volume_limit_factor": 15,  # Factor to multiply opposite volume by for our quantity limit
                                     # 1.0 means limit to exact opposite volume
                                     # > 1.0 allows placing more, < 1.0 less
                                     # Set to None or a very large number to disable volume limit
    },
    Product.SQUID_INK: {
        "fair_value": 2016,  # Example fair value, could be dynamic
        "take_width": 1,      # How far from fair value to take aggressive orders
        "clear_width": 2,     # How far from fair value to clear inventory aggressively
        # for making
        "disregard_edge": 1,  # Don't penny/join levels this close to fair value
        "join_edge": 5,       # Join levels within this edge DEFAULT was 2
        "default_edge": 4,    # Default distance from fair value to place orders
        "soft_position_limit": 50,  # Adjust price slightly if position exceeds this
        "volume_limit_factor": 15,  # Factor to multiply opposite volume by for our quantity limit
                                     # 1.0 means limit to exact opposite volume
                                     # > 1.0 allows placing more, < 1.0 less
                                     # Set to None or a very large number to disable volume limit
    },

    Product.CROISSANTS : {
        # "fair_value" : ,
        "soft_position_limit" :  250, 
    },
    Product.JAMS :{
        "fair_value" : 6540, 
        "soft_position_limit" :  350, 
    },
    Product.DJEMBES :{
       "fair_value" : 13450,
         "position_limit": 60,
       "exit_reference_ema_period": 20, # EMA to gauge movement against the trend
        "scale_out_factor": 30, 
    },
    Product.BASKET1 :{
        # "fair_value" : ,
        "soft_position_limit" :  60, 
    },
    Product.BASKET2 :{
        # "fair_value" : ,
        "soft_position_limit" :  100, 
    },

     Product.SPREAD: {
        "default_spread_mean": 10,
        "default_spread_std": 114.01932291969442,
        "spread_std_window": 75,
        "zscore_threshold": 2.5,
        "target_position": 60,
    },
    Product.SPREAD2: {
        "default_spread_mean": 22.56946666666666,
        "default_spread_std": 57.91713972894571,
        "spread_std_window": 75,
        "zscore_threshold": 2.75,
        "target_position": 100,
    }
    ,
    Product.BASKET1: {
        "default_spread_mean": 48.762433333333334 ,
        "default_spread_std": 85.1180321401536,
        "spread_std_window": 45,
        "zscore_threshold": 1,
        "target_position": 60,
    },
    Product.BASKET2: {
        "default_spread_mean": 30.235966666666666,
        "default_spread_std": 59.84820272766946,
        "spread_std_window": 45,
        "zscore_threshold": 7,
        "target_position": 60,
    },
    Product.MAGNIFICENT_MACARONS: {
        "make_edge": 1,
        "make_min_edge": 0.5,
        "make_probability": 0.5,
        "init_make_edge": -1.5, # This seems unusual (negative edge?), maybe it's adaptive starting point? Retaining for now.
        "min_edge": 0.6,
        "volume_avg_timestamp": 3,
        "volume_bar": 50,
        "dec_edge_discount": 0.6,
        "step_size": 0.3,
        "csi": 46,  # Critical Sunlight Index determined from analysis
        "low_sun_aggr_buy_vol": 15, # Max volume per order when buying in low sun regime (increased from 10)
        "low_sun_aggr_price_offset": 1, # How many ticks above best ask to place buy orders in low sun
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


    
    def get_swmid(self, order_depth) -> float:
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_vol = abs(order_depth.buy_orders[best_bid])
        best_ask_vol = abs(order_depth.sell_orders[best_ask])
        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (
            best_bid_vol + best_ask_vol
        )

    def get_synthetic_basket_order_depth(
        self, order_depths: Dict[str, OrderDepth]
    ) -> OrderDepth:
        # Constants
        JAMS_PER_BASKET = BASKET_1_WEIGHTS[Product.JAMS]
        CROISSANTS_PER_BASKET = BASKET_1_WEIGHTS[Product.CROISSANTS]
        DJEMBES_PER_BASKET = BASKET_1_WEIGHTS[Product.DJEMBES]

        # Initialize the synthetic basket order depth
        synthetic_order_price = OrderDepth()

        # Calculate the best bid and ask for each component
        jams_best_bid = (
            max(order_depths[Product.JAMS].buy_orders.keys())
            if order_depths[Product.JAMS].buy_orders
            else 0
        )
        jams_best_ask = (
            min(order_depths[Product.JAMS].sell_orders.keys())
            if order_depths[Product.JAMS].sell_orders
            else float("inf")
        )
        croissants_best_bid = (
            max(order_depths[Product.CROISSANTS].buy_orders.keys())
            if order_depths[Product.CROISSANTS].buy_orders
            else 0
        )
        croissants_best_ask = (
            min(order_depths[Product.CROISSANTS].sell_orders.keys())
            if order_depths[Product.CROISSANTS].sell_orders
            else float("inf")
        )
        djembes_best_bid = (
            max(order_depths[Product.DJEMBES].buy_orders.keys())
            if order_depths[Product.DJEMBES].buy_orders
            else 0
        )
        djembes_best_ask = (
            min(order_depths[Product.DJEMBES].sell_orders.keys())
            if order_depths[Product.DJEMBES].sell_orders
            else float("inf")
        )

        # Calculate the implied bid and ask for the synthetic basket
        implied_bid = (
            jams_best_bid * JAMS_PER_BASKET
            + croissants_best_bid * CROISSANTS_PER_BASKET
            + djembes_best_bid * DJEMBES_PER_BASKET
        )
        implied_ask = (
            jams_best_ask * JAMS_PER_BASKET
            + croissants_best_ask * CROISSANTS_PER_BASKET
            + djembes_best_ask * DJEMBES_PER_BASKET
        )

        # Calculate the maximum number of synthetic baskets available at the implied bid and ask
        if implied_bid > 0:
            jams_bid_volume = (
                order_depths[Product.JAMS].buy_orders[jams_best_bid]
                // JAMS_PER_BASKET
            )
            croissants_bid_volume = (
                order_depths[Product.CROISSANTS].buy_orders[croissants_best_bid]
                // CROISSANTS_PER_BASKET
            )
            djembes_bid_volume = (
                order_depths[Product.DJEMBES].buy_orders[djembes_best_bid]
                // DJEMBES_PER_BASKET
            )
            implied_bid_volume = min(
                jams_bid_volume, croissants_bid_volume, djembes_bid_volume
            )
            synthetic_order_price.buy_orders[implied_bid] = implied_bid_volume

        if implied_ask < float("inf"):
            jams_ask_volume = (
                -order_depths[Product.JAMS].sell_orders[jams_best_ask]
                // JAMS_PER_BASKET
            )
            croissants_ask_volume = (
                -order_depths[Product.CROISSANTS].sell_orders[croissants_best_ask]
                // CROISSANTS_PER_BASKET
            )
            djembes_ask_volume = (
                -order_depths[Product.DJEMBES].sell_orders[djembes_best_ask]
                // DJEMBES_PER_BASKET
            )
            implied_ask_volume = min(
                jams_ask_volume, croissants_ask_volume, djembes_ask_volume
            )
            synthetic_order_price.sell_orders[implied_ask] = -implied_ask_volume

        return synthetic_order_price

    def convert_synthetic_basket_orders(
        self, synthetic_orders: List[Order], order_depths: Dict[str, OrderDepth]
    ) -> Dict[str, List[Order]]:
        # Initialize the dictionary to store component orders
        component_orders = {
            Product.JAMS: [],
            Product.CROISSANTS: [],
            Product.DJEMBES: [],
        }

        # Get the best bid and ask for the synthetic basket
        synthetic_basket_order_depth = self.get_synthetic_basket_order_depth(
            order_depths
        )
        best_bid = (
            max(synthetic_basket_order_depth.buy_orders.keys())
            if synthetic_basket_order_depth.buy_orders
            else 0
        )
        best_ask = (
            min(synthetic_basket_order_depth.sell_orders.keys())
            if synthetic_basket_order_depth.sell_orders
            else float("inf")
        )

        # Iterate through each synthetic basket order
        for order in synthetic_orders:
            # Extract the price and quantity from the synthetic basket order
            price = order.price
            quantity = order.quantity

            # Check if the synthetic basket order aligns with the best bid or ask
            if quantity > 0 and price >= best_ask:
                # Buy order - trade components at their best ask prices
                jams_price = min(
                    order_depths[Product.JAMS].sell_orders.keys()
                )
                croissants_price = min(
                    order_depths[Product.CROISSANTS].sell_orders.keys()
                )
                djembes_price = min(order_depths[Product.DJEMBES].sell_orders.keys())
            elif quantity < 0 and price <= best_bid:
                # Sell order - trade components at their best bid prices
                jams_price = max(order_depths[Product.JAMS].buy_orders.keys())
                croissants_price = max(
                    order_depths[Product.CROISSANTS].buy_orders.keys()
                )
                djembes_price = max(order_depths[Product.DJEMBES].buy_orders.keys())
            else:
                # The synthetic basket order does not align with the best bid or ask
                continue

            # Create orders for each component
            jams_order = Order(
                Product.JAMS,
                jams_price,
                quantity * BASKET_1_WEIGHTS[Product.JAMS],
            )
            croissants_order = Order(
                Product.CROISSANTS,
                croissants_price,
                quantity * BASKET_1_WEIGHTS[Product.CROISSANTS],
            )
            djembes_order = Order(
                Product.DJEMBES, djembes_price, quantity * BASKET_1_WEIGHTS[Product.DJEMBES]
            )

            # Add the component orders to the respective lists
            component_orders[Product.JAMS].append(jams_order)
            component_orders[Product.CROISSANTS].append(croissants_order)
            component_orders[Product.DJEMBES].append(djembes_order)

        return component_orders

    def execute_spread_orders(
        self,
        target_position: int,
        basket_position: int,
        order_depths: Dict[str, OrderDepth],
    ):

        if target_position == basket_position:
            return None

        target_quantity = abs(target_position - basket_position)
        basket_order_depth = order_depths[Product.BASKET1]
        synthetic_order_depth = self.get_synthetic_basket_order_depth(order_depths)

        if target_position > basket_position:
            basket_ask_price = min(basket_order_depth.sell_orders.keys())
            basket_ask_volume = abs(basket_order_depth.sell_orders[basket_ask_price])

            synthetic_bid_price = max(synthetic_order_depth.buy_orders.keys())
            synthetic_bid_volume = abs(
                synthetic_order_depth.buy_orders[synthetic_bid_price]
            )

            orderbook_volume = min(basket_ask_volume, synthetic_bid_volume)
            execute_volume = min(orderbook_volume, target_quantity)

            basket_orders = [
                Order(Product.BASKET1, basket_ask_price, execute_volume)
            ]
            synthetic_orders = [
                Order(Product.SYNTHETIC, synthetic_bid_price, -execute_volume)
            ]

            aggregate_orders = self.convert_synthetic_basket_orders(
                synthetic_orders, order_depths
            )
            aggregate_orders[Product.BASKET1] = basket_orders
            return aggregate_orders

        else:
            basket_bid_price = max(basket_order_depth.buy_orders.keys())
            basket_bid_volume = abs(basket_order_depth.buy_orders[basket_bid_price])

            synthetic_ask_price = min(synthetic_order_depth.sell_orders.keys())
            synthetic_ask_volume = abs(
                synthetic_order_depth.sell_orders[synthetic_ask_price]
            )

            orderbook_volume = min(basket_bid_volume, synthetic_ask_volume)
            execute_volume = min(orderbook_volume, target_quantity)

            basket_orders = [
                Order(Product.BASKET1, basket_bid_price, -execute_volume)
            ]
            synthetic_orders = [
                Order(Product.SYNTHETIC, synthetic_ask_price, execute_volume)
            ]

            aggregate_orders = self.convert_synthetic_basket_orders(
                synthetic_orders, order_depths
            )
            aggregate_orders[Product.BASKET1] = basket_orders
            return aggregate_orders

    def spread_orders(
        self,
        order_depths: Dict[str, OrderDepth],
        product: Product,
        basket_position: int,
        spread_data: Dict[str, Any],
    ):
        if Product.BASKET1 not in order_depths.keys():
            return None

        basket_order_depth = order_depths[Product.BASKET1]
        synthetic_order_depth = self.get_synthetic_basket_order_depth(order_depths)
        basket_swmid = self.get_swmid(basket_order_depth)
        synthetic_swmid = self.get_swmid(synthetic_order_depth)
        spread = basket_swmid - synthetic_swmid
        spread_data["spread_history"].append(spread)

        if (
            len(spread_data["spread_history"])
            < self.params[Product.SPREAD]["spread_std_window"]
        ):
            return None
        elif len(spread_data["spread_history"]) > self.params[Product.SPREAD]["spread_std_window"]:
            spread_data["spread_history"].pop(0)

        spread_std = np.std(spread_data["spread_history"])

        zscore = (
            spread - self.params[Product.SPREAD]["default_spread_mean"]
        ) / spread_std

        if zscore >= self.params[Product.SPREAD]["zscore_threshold"]:
            if basket_position != -self.params[Product.SPREAD]["target_position"]:
                return self.execute_spread_orders(
                    -self.params[Product.SPREAD]["target_position"],
                    basket_position,
                    order_depths,
                )

        if zscore <= -self.params[Product.SPREAD]["zscore_threshold"]:
            if basket_position != self.params[Product.SPREAD]["target_position"]:
                return self.execute_spread_orders(
                    self.params[Product.SPREAD]["target_position"],
                    basket_position,
                    order_depths,
                )

        spread_data["prev_zscore"] = zscore
        return None


    def get_synthetic_basket_order_depth_2(
        self, order_depths: Dict[str, OrderDepth]
    ) -> OrderDepth:
        # Constants
        JAMS_PER_BASKET = BASKET_2_WEIGHTS[Product.JAMS]
        CROISSANTS_PER_BASKET = BASKET_2_WEIGHTS[Product.CROISSANTS]

        # Initialize the synthetic basket order depth
        synthetic_order_price = OrderDepth()

        # Calculate the best bid and ask for each component
        jams_best_bid = (
            max(order_depths[Product.JAMS].buy_orders.keys())
            if order_depths[Product.JAMS].buy_orders
            else 0
        )
        jams_best_ask = (
            min(order_depths[Product.JAMS].sell_orders.keys())
            if order_depths[Product.JAMS].sell_orders
            else float("inf")
        )
        croissants_best_bid = (
            max(order_depths[Product.CROISSANTS].buy_orders.keys())
            if order_depths[Product.CROISSANTS].buy_orders
            else 0
        )
        croissants_best_ask = (
            min(order_depths[Product.CROISSANTS].sell_orders.keys())
            if order_depths[Product.CROISSANTS].sell_orders
            else float("inf")
        )
        # Calculate the implied bid and ask for the synthetic basket
        implied_bid = (
            jams_best_bid * JAMS_PER_BASKET
            + croissants_best_bid * CROISSANTS_PER_BASKET
        )
        implied_ask = (
            jams_best_ask * JAMS_PER_BASKET
            + croissants_best_ask * CROISSANTS_PER_BASKET
        )

        # Calculate the maximum number of synthetic baskets available at the implied bid and ask
        if implied_bid > 0:
            jams_bid_volume = (
                order_depths[Product.JAMS].buy_orders[jams_best_bid]
                // JAMS_PER_BASKET
            )
            croissants_bid_volume = (
                order_depths[Product.CROISSANTS].buy_orders[croissants_best_bid]
                // CROISSANTS_PER_BASKET
            )
            implied_bid_volume = min(
                jams_bid_volume, croissants_bid_volume
            )
            synthetic_order_price.buy_orders[implied_bid] = implied_bid_volume

        if implied_ask < float("inf"):
            jams_ask_volume = (
                -order_depths[Product.JAMS].sell_orders[jams_best_ask]
                // JAMS_PER_BASKET
            )
            croissants_ask_volume = (
                -order_depths[Product.CROISSANTS].sell_orders[croissants_best_ask]
                // CROISSANTS_PER_BASKET
            )
            implied_ask_volume = min(
                jams_ask_volume, croissants_ask_volume
            )
            synthetic_order_price.sell_orders[implied_ask] = -implied_ask_volume

        return synthetic_order_price

    def convert_synthetic_basket_orders_2(
        self, synthetic_orders: List[Order], order_depths: Dict[str, OrderDepth]
    ) -> Dict[str, List[Order]]:
        # Initialize the dictionary to store component orders
        component_orders = {
            Product.JAMS: [],
            Product.CROISSANTS: [],
        }

        # Get the best bid and ask for the synthetic basket
        synthetic_basket_order_depth = self.get_synthetic_basket_order_depth_2(
            order_depths
        )
        best_bid = (
            max(synthetic_basket_order_depth.buy_orders.keys())
            if synthetic_basket_order_depth.buy_orders
            else 0
        )
        best_ask = (
            min(synthetic_basket_order_depth.sell_orders.keys())
            if synthetic_basket_order_depth.sell_orders
            else float("inf")
        )

        # Iterate through each synthetic basket order
        for order in synthetic_orders:
            # Extract the price and quantity from the synthetic basket order
            price = order.price
            quantity = order.quantity

            # Check if the synthetic basket order aligns with the best bid or ask
            if quantity > 0 and price >= best_ask:
                # Buy order - trade components at their best ask prices
                jams_price = min(
                    order_depths[Product.JAMS].sell_orders.keys()
                )
                croissants_price = min(
                    order_depths[Product.CROISSANTS].sell_orders.keys()
                )
            elif quantity < 0 and price <= best_bid:
                # Sell order - trade components at their best bid prices
                jams_price = max(order_depths[Product.JAMS].buy_orders.keys())
                croissants_price = max(
                    order_depths[Product.CROISSANTS].buy_orders.keys()
                )
            else:
                # The synthetic basket order does not align with the best bid or ask
                continue

            # Create orders for each component
            jams_order = Order(
                Product.JAMS,
                jams_price,
                quantity * BASKET_2_WEIGHTS[Product.JAMS],
            )
            croissants_order = Order(
                Product.CROISSANTS,
                croissants_price,
                quantity * BASKET_2_WEIGHTS[Product.CROISSANTS],
            )
            # Add the component orders to the respective lists
            component_orders[Product.JAMS].append(jams_order)
            component_orders[Product.CROISSANTS].append(croissants_order)

        return component_orders

    def execute_spread_orders_2(
        self,
        target_position: int,
        basket_position: int,
        order_depths: Dict[str, OrderDepth],
    ):

        if target_position == basket_position:
            return None

        target_quantity = abs(target_position - basket_position)
        basket_order_depth = order_depths[Product.BASKET2]
        synthetic_order_depth = self.get_synthetic_basket_order_depth_2(order_depths)

        if target_position > basket_position:
            basket_ask_price = min(basket_order_depth.sell_orders.keys())
            basket_ask_volume = abs(basket_order_depth.sell_orders[basket_ask_price])

            synthetic_bid_price = max(synthetic_order_depth.buy_orders.keys())
            synthetic_bid_volume = abs(
                synthetic_order_depth.buy_orders[synthetic_bid_price]
            )

            orderbook_volume = min(basket_ask_volume, synthetic_bid_volume)
            execute_volume = min(orderbook_volume, target_quantity)

            basket_orders = [
                Order(Product.BASKET2, basket_ask_price, execute_volume)
            ]
            synthetic_orders = [
                Order(Product.SYNTHETIC2, synthetic_bid_price, -execute_volume)
            ]

            aggregate_orders = self.convert_synthetic_basket_orders_2(
                synthetic_orders, order_depths
            )
            aggregate_orders[Product.BASKET2] = basket_orders
            return aggregate_orders

        else:
            basket_bid_price = max(basket_order_depth.buy_orders.keys())
            basket_bid_volume = abs(basket_order_depth.buy_orders[basket_bid_price])

            synthetic_ask_price = min(synthetic_order_depth.sell_orders.keys())
            synthetic_ask_volume = abs(
                synthetic_order_depth.sell_orders[synthetic_ask_price]
            )

            orderbook_volume = min(basket_bid_volume, synthetic_ask_volume)
            execute_volume = min(orderbook_volume, target_quantity)

            basket_orders = [
                Order(Product.BASKET2, basket_bid_price, -execute_volume)
            ]
            synthetic_orders = [
                Order(Product.SYNTHETIC2, synthetic_ask_price, execute_volume)
            ]

            aggregate_orders = self.convert_synthetic_basket_orders_2(
                synthetic_orders, order_depths
            )
            aggregate_orders[Product.BASKET2] = basket_orders
            return aggregate_orders

    def spread_orders_2(
        self,
        order_depths: Dict[str, OrderDepth],
        product: Product,
        basket_position: int,
        spread_data: Dict[str, Any],
    ):
        if Product.BASKET2 not in order_depths.keys():
            return None

        basket_order_depth = order_depths[Product.BASKET2]
        synthetic_order_depth = self.get_synthetic_basket_order_depth_2(order_depths)
        basket_swmid = self.get_swmid(basket_order_depth)
        synthetic_swmid = self.get_swmid(synthetic_order_depth)
        spread = basket_swmid - synthetic_swmid
        spread_data["spread_history"].append(spread)

        if (
            len(spread_data["spread_history"])
            < self.params[Product.SPREAD2]["spread_std_window"]
        ):
            return None
        elif len(spread_data["spread_history"]) > self.params[Product.SPREAD2]["spread_std_window"]:
            spread_data["spread_history"].pop(0)

        spread_std = np.std(spread_data["spread_history"])
        spread_data["curr_len"] += 1
        spread_data["inc_mean"] = spread_data["inc_mean"] + (spread - spread_data["inc_mean"]) / spread_data["curr_len"]
        zscore = (spread - spread_data["inc_mean"]) / spread_std

        # zscore = (
        #     spread - self.params[Product.SPREAD2]["default_spread_mean"]
        # ) / spread_std


        if zscore >= self.params[Product.SPREAD2]["zscore_threshold"]:
            if basket_position != -self.params[Product.SPREAD2]["target_position"]:
                return self.execute_spread_orders_2(
                    -self.params[Product.SPREAD2]["target_position"],
                    basket_position,
                    order_depths,
                )

        if zscore <= -self.params[Product.SPREAD2]["zscore_threshold"]:
            if basket_position != self.params[Product.SPREAD2]["target_position"]:
                return self.execute_spread_orders_2(
                    self.params[Product.SPREAD2]["target_position"],
                    basket_position,
                    order_depths,
                )

        spread_data["prev_zscore"] = zscore
        return None
    
    def save_prices(self, state: TradingState):
        CROISSANTS = Product.CROISSANTS
        JAMS = Product.JAMS
        DJEMBES = Product.DJEMBES
        price_croissants = self.get_mid_price(CROISSANTS, state)
        price_jams = self.get_mid_price(JAMS, state)
        timestamp = state.timestamp

        self.prices[CROISSANTS] = pd.concat([
            self.prices[CROISSANTS],
            pd.Series({timestamp: price_croissants}, dtype=float)
        ])

        self.prices[JAMS] = pd.concat([
            self.prices[JAMS],
            pd.Series({timestamp: price_jams}, dtype=float)
        ])

        # spread_value = price_jams - 0.65182388 * price_croissants
        # spread_value = price_jams - 0.65362656 * price_croissants
        spread_value = price_jams - 1.52*price_croissants
        self.prices["Spread"] = pd.concat([
            self.prices["Spread"],
            pd.Series({timestamp: spread_value}, dtype=float)
        ])
    
    def save_prices_2(self, state: TradingState):
        CROISSANTS = Product.CROISSANTS
        JAMS = Product.JAMS
        DJEMBES = Product.DJEMBES
        price_croissants = self.get_mid_price(CROISSANTS, state)
        price_jams = self.get_mid_price(DJEMBES, state)
        timestamp = state.timestamp

        self.prices[CROISSANTS] = pd.concat([
            self.prices[CROISSANTS],
            pd.Series({timestamp: price_croissants}, dtype=float)
        ])

        self.prices[DJEMBES] = pd.concat([
            self.prices[DJEMBES],
            pd.Series({timestamp: price_jams}, dtype=float)
        ])

        # spread_value = price_jams - 3.132 * price_croissants
        spread_value = price_jams - 0.319 * price_croissants
        self.prices["Spread2"] = pd.concat([
            self.prices["Spread2"],
            pd.Series({timestamp: spread_value}, dtype=float)
        ])
        # self.prices["Spread"].append(price_jams - 0.65182388*price_croissants)

    def pair_strategy(self, state: TradingState):
        orders_croissants = []
        orders_jams = []

        self.save_prices(state)

        if len(self.prices["Spread"]) < WINDOW:
            # logger.print(f"Not enough data for window {WINDOW}, have {len(self.prices['Spread'])}")
            return orders_croissants, orders_jams

        mid_price_croissants = self.get_mid_price(Product.CROISSANTS, state)
        mid_price_jams = self.get_mid_price(Product.JAMS, state)

        int_price_croissants = int(mid_price_croissants)
        int_price_jams = int(mid_price_jams)

        croissants_position = self.get_position(Product.CROISSANTS, state)
        # jams_position = self.get_position(Product.JAMS, state)

        avg_spread = self.prices["Spread"].rolling(WINDOW).mean()
        std_spread = self.prices["Spread"].rolling(WINDOW).std()
        spread_short = self.prices["Spread"].rolling(5).mean()

        avg_spread = avg_spread.iloc[-1]
        std_spread = std_spread.iloc[-1]
        spread_short = spread_short.iloc[-1]

        # logger.print(spread_short)

        logger.print(f"Spread: {spread_short}, avg: {avg_spread}, std: {std_spread}")

        # - croissants -coconuts
        price_adj = 0
        threshold = 1.75
        ORDER_VOLUME = 20

        if abs(croissants_position) < POSITION_LIMITS[Product.CROISSANTS] - 20:
            if spread_short < avg_spread - threshold*std_spread:
                orders_croissants.append(Order(Product.CROISSANTS, int_price_croissants-price_adj, -ORDER_VOLUME))
                orders_jams.append(Order(Product.JAMS, int_price_jams+price_adj, (int)(ORDER_VOLUME*1.4)))


            elif spread_short > avg_spread + threshold*std_spread:
                orders_croissants.append(Order(Product.CROISSANTS, int_price_croissants+price_adj, ORDER_VOLUME))
                orders_jams.append(Order(Product.JAMS, int_price_jams-price_adj, (int)(-ORDER_VOLUME*1.4)))

        else :
            if croissants_position > 0:
                if spread_short < avg_spread - threshold*std_spread:
                    orders_croissants.append(Order(Product.CROISSANTS, int_price_croissants-price_adj, -ORDER_VOLUME))
                    orders_jams.append(Order(Product.JAMS, int_price_jams+price_adj, (int)(ORDER_VOLUME*1.4)))

            else :
                if spread_short > avg_spread + threshold*std_spread:
                    orders_croissants.append(Order(Product.CROISSANTS, int_price_croissants+price_adj, ORDER_VOLUME))
                    orders_jams.append(Order(Product.JAMS, int_price_jams-price_adj, (int)(-ORDER_VOLUME*1.4)))


        return orders_croissants, orders_jams

    


        


    def resin_strategy(self , state:TradingState):
        # --- RAINFOREST_RESIN Logic ---
        product = Product.RAINFOREST_RESIN
        # result = {} # No longer need this dictionary here
        orders = [] # Initialize an empty list for orders

        if product in self.params and product in state.order_depths:
            order_depth = state.order_depths[product]
            position = state.position.get(product, 0)
            params = self.params[product]

            resin_take_orders, buy_vol_after_take, sell_vol_after_take = self.take_orders(
                product, order_depth, params["fair_value"], params["take_width"], position
            )

            resin_clear_orders, buy_vol_after_clear, sell_vol_after_clear = self.clear_orders(
                product, order_depth, params["fair_value"], params["clear_width"],
                position, buy_vol_after_take, sell_vol_after_take
            )

            resin_make_orders, _, _ = self.make_orders(
                product,
                order_depth,
                params["fair_value"],
                position,
                buy_vol_after_clear,
                sell_vol_after_clear,
                params,
                manage_position=True
            )

            # Combine the orders into a single list
            orders = resin_take_orders + resin_clear_orders + resin_make_orders
           
        # Return the list of orders directly (or an empty list if the 'if' was false)
        return orders
    
    def get_weighted_mid_price(self, product, state: TradingState ):
        default_price = Product.ema_prices.get[product] or 2045
        if product not in state.order_depths:
            return default_price

        order_depth = state.order_depths[product]
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return default_price # Fallback to simple mid or default needed

        # Get best bid/ask and their volumes
        best_bid, best_bid_vol = max(order_depth.buy_orders.items())
        best_ask, best_ask_vol = min(order_depth.sell_orders.items())

        # Ensure volumes are positive
        best_bid_vol = abs(best_bid_vol)
        best_ask_vol = abs(best_ask_vol)

        # Calculate weighted mid-price
        total_vol = best_bid_vol + best_ask_vol
        if total_vol == 0:
            return (best_bid + best_ask) / 2 # Avoid division by zero, fallback to simple mid

        weighted_mid = (best_bid * best_ask_vol + best_ask * best_bid_vol) / total_vol
        return weighted_mid
    

    # NEW strategy for KELP
    def market_make_kelp(self, state: TradingState ):
        """Places bid and ask orders around the EMA fair value for KELP."""
        orders = []
        product = Product.KELP
        position_product = self.get_position(product, state)

        # Use EMA as the fair value estimate
        fair_value = Product.ema_prices.get(product, None)
        if fair_value is None:
             # Attempt to calculate current mid-price if EMA isn't ready
            fair_value = self.get_weighted_mid_price(product, state)
            if fair_value is None: # Still None? Use default.
                 fair_value = 2045


        # Define our target bid and ask prices
        buy_price = round_price(fair_value - self.kelp_spread_margin)
        sell_price = round_price(fair_value + self.kelp_spread_margin)

        # Ensure buy_price is strictly less than sell_price
        if buy_price >= sell_price:
             return [] # Avoid placing overlapping or inverted orders


        # Calculate order volumes based on position limits
        # Buy orders: positive quantity
        bid_volume = 50 - position_product
        # Sell orders: negative quantity
        ask_volume = -(50 + position_product) # Note the negative sign

        # Place orders only if we have volume to trade
        if bid_volume > 0:
            orders.append(Order(product, buy_price, bid_volume))
          
        if ask_volume < 0: # Check for negative volume for selling
            orders.append(Order(product, sell_price, ask_volume))
            logger.print(f"Placing KELP Sell: {ask_volume} @ {sell_price}")

        return orders
    

    def market_making_range_bound(self, state: TradingState) -> List[Order]:
        product = Product.JAMS
        orders: List[Order] = []
        order_depth: OrderDepth = state.order_depths.get(product, None)

        if not order_depth or not order_depth.buy_orders or not order_depth.sell_orders:
            return orders

        best_bid: float = max(order_depth.buy_orders.keys())
        best_ask: float = min(order_depth.sell_orders.keys())
        mid_price: float = (best_bid + best_ask) / 2

        # Stop-loss and Profit Target logic
        position: int = self.position[product]  # Use the class-level position
        if self.entry_price[product] is not None:
            profit = mid_price - self.entry_price[product] if position > 0 else self.entry_price[product] - mid_price

            if position > 0 and profit >= self.profit_target:
                # Profit target reached for long position
                orders.append(Order(product, best_bid, -position))
                self.position[product] = 0  # Reset position
                self.entry_price[product] = None
                # logging.info(f"Profit target (long). Closed at {best_bid}, Profit: {profit}")


            elif position < 0 and profit >= self.profit_target:
                # Profit target reached for short position
                orders.append(Order(product, best_ask, -position))
                self.position[product] = 0  # Reset position
                self.entry_price[product] = None
                # logging.info(f"Profit target (short). Closed at {best_ask}, Profit: {profit}")

            elif position > 0 and profit <= -self.stop_loss:
                # Stop-loss triggered for long position
                orders.append(Order(product, best_bid, -position))
                self.position[product] = 0  # Reset position
                self.entry_price[product] = None
                # logging.info(f"Stop-loss (long). Closed at {best_bid}, Loss: {profit}")

            elif position < 0 and profit <= -self.stop_loss:
                # Stop-loss triggered for short position
                orders.append(Order(product, best_ask, -position))
                self.position[product] = 0  # Reset position
                self.entry_price[product] = None
                # logging.info(f"Stop-loss (short). Closed at {best_ask}, Loss: {profit}")


        # Calculate target buy and sell prices
        buy_price = mid_price - (self.k_factor * mid_price)
        sell_price = mid_price + (self.k_factor * mid_price)

        # Place buy order if price within range and we are not already long at our limit
        if self.price_min <= buy_price <= self.price_max and position < self.LIMIT[product]:
            desired_trade = self.LIMIT[product] - position
            order_size = min(desired_trade, self.order_size)
            # Use min to ensure we don't exceed the maximum order size allowed
            orders.append(Order(product, int(buy_price), int(order_size)))
            if position == 0:
                self.entry_price[product] = int(buy_price)
            self.position[product] += int(order_size)
            # logging.info(f"Buying {order_size} {product} at {buy_price}")

        # Place sell order if price within range and we are not already short at our limit
        if self.price_min <= sell_price <= self.price_max and position > -self.LIMIT[product]:
            desired_trade = -self.LIMIT[product] - position
            order_size = max(desired_trade, -self.order_size)
            # Use max to ensure we don't exceed the maximum order size allowed
            orders.append(Order(product, int(sell_price), int(order_size)))

            if position == 0:
                self.entry_price[product] = int(sell_price)

            self.position[product] += int(order_size)
            # logging.info(f"Selling {abs(order_size)} {product} at {sell_price}")

        return orders

    
    def norm_cdf(self, x: float) -> float:
        """Approximate the cumulative distribution function for standard normal."""
        if x < 0:
            return 1 - self.norm_cdf(-x)
        t = 1 / (1 + 0.2316419 * x)
        coefficients = [0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429]
        sum_terms = 0
        for i, coef in enumerate(coefficients):
            sum_terms += coef * (t ** (i + 1))
        return 1 - (1 / math.sqrt(2 * math.pi)) * math.exp(-x**2 / 2) * sum_terms

    def black_scholes_call(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate Black-Scholes call option price."""
        S=S-2
        
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return max(0, S - K)  # Intrinsic value at expiry or invalid inputs
        try:
            d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
            d2 = d1 - sigma * math.sqrt(T)
            call_price = S * self.norm_cdf(d1) - K * math.exp(-r * T) * self.norm_cdf(d2)
            return max(0, call_price)
        except (ValueError, ZeroDivisionError):
            return max(0, S - K)  # Fallback to intrinsic value

    def estimate_volatility(self, prices: List[float], window_size: int = 3) -> float:
        """Estimate historical volatility from price history with a specified window."""
        if len(prices) < 2:
            return 0.3  # Default volatility (30% annual)
    # Use the last window_size prices, or all if fewer are available
        recent_prices = prices[-window_size:] if len(prices) >= window_size else prices
        if len(recent_prices) < 2:
            return 0.3
        log_returns = [math.log(recent_prices[i] / recent_prices[i-1]) for i in range(1, len(recent_prices))]
        mean_return = sum(log_returns) / len(log_returns)
        variance = sum((ret - mean_return)**2 for ret in log_returns) / max(1, len(log_returns) - 1)
        volatility = math.sqrt(variance) * math.sqrt(252) if variance > 0 else 0.3
        return max(volatility, 0.1)  # Ensure minimum volatility of 10%

    def get_position(self, product, state: TradingState):
        return state.position.get(product, 0)


    def get_value_on_product(self, product, state: TradingState):
        return self.get_position(product, state) * self.get_mid_price(product, state)

    
    def update_pnl(self, state: TradingState):
        def update_cash():
            for product in state.own_trades:
                for trade in state.own_trades[product]:
                    if trade.timestamp != state.timestamp - 100:
                        continue
                    if trade.buyer == SUBMISSION:
                        self.cash -= trade.quantity * trade.price
                    if trade.seller == SUBMISSION:
                        self.cash += trade.quantity * trade.price

        def get_value_on_positions():
            return sum(self.get_value_on_product(p, state) for p in state.position)

        update_cash()
        return self.cash + get_value_on_positions()
    
    def update_ema_prices(self, state: TradingState):
       for product in [Product.RAINFOREST_RESIN, Product.KELP, Product.SQUID_INK , Product.BASKET1 , Product.BASKET2 , Product.CROISSANTS , Product.DJEMBES]:
            if not isinstance(product, str):
                continue  
            mid_price = self.get_mid_price(product, state)
            if mid_price is None:
                continue

            if Product.ema_prices.get(product) is None:
                Product.ema_prices[product] = mid_price
            else:
                Product.ema_prices[product] = (
                    Product.ema_param* mid_price +
                    (1 - Product.ema_param) * Product.ema_prices[product]
                )

    def update_past_prices(self, state: TradingState) -> None:
       for product in [Product.RAINFOREST_RESIN, Product.KELP, Product.SQUID_INK , Product.BASKET1 , Product.BASKET2 , Product.CROISSANTS , Product.DJEMBES , 'VOLCANIC_ROCK']:

            if not isinstance(product, str):
                continue
            mid_price = self.get_mid_price(product, state)
            if mid_price is not None:
                Product.past_prices[product].append(mid_price)
    def take_best_orders(
        self, product: str, fair_value: int, take_width: float, orders: List[Order],
        order_depth: OrderDepth, position: int, buy_order_volume: int, sell_order_volume: int,
        prevent_adverse: bool = False, adverse_volume: int = 0,
    ) -> (int, int):
        position_limit = self.LIMIT[product]
        # Buy Aggression (Hit Ask)
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = abs(order_depth.sell_orders[best_ask])  # Positive volume
            if not prevent_adverse or best_ask_amount <= adverse_volume:
                if best_ask <= fair_value - take_width:
                    quantity = min(best_ask_amount, position_limit - position - buy_order_volume)  # Max we can buy
                    if quantity > 0:

                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
        # Sell Aggression (Hit Bid)
        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]  # Positive volume
            if not prevent_adverse or best_bid_amount <= adverse_volume:
                if best_bid >= fair_value + take_width:
                    quantity = min(best_bid_amount, position_limit + position - sell_order_volume)  # Max we can sell
                    if quantity > 0:
                        
                        orders.append(Order(product, best_bid, -quantity))
                        sell_order_volume += quantity
        return buy_order_volume, sell_order_volume

    def clear_position_order(
        self, product: str, fair_value: float, width: int, orders: List[Order],
        order_depth: OrderDepth, position: int, buy_order_volume: int, sell_order_volume: int,
    ) -> (int, int):
        position_limit = self.LIMIT[product]
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)  # Price to place clearing buy
        fair_for_ask = round(fair_value + width)  # Price to place clearing sell

        # Calculate remaining capacity *after* takes and potential clears
        remaining_buy_capacity = position_limit - (position + buy_order_volume)
        remaining_sell_capacity = position_limit + (position - sell_order_volume)

        # If long, try to sell aggressively to clear
        if position_after_take > 0:
            clear_ask_price = fair_for_ask  # Place sell at this price or better
            volume_to_hit = 0
            for price, volume in sorted(order_depth.buy_orders.items(), reverse=True):
                if price >= clear_ask_price:
                    volume_to_hit += volume
                else:
                    break
            qty_to_clear = min(position_after_take, volume_to_hit)
            sent_quantity = min(remaining_sell_capacity, qty_to_clear)
            if sent_quantity > 0:
                orders.append(Order(product, clear_ask_price, -sent_quantity))
                sell_order_volume += sent_quantity

        # If short, try to buy aggressively to clear
        if position_after_take < 0:
            clear_bid_price = fair_for_bid  # Place buy at this price or better
            volume_to_hit = 0
            for price, volume in sorted(order_depth.sell_orders.items()):
                if price <= clear_bid_price:
                    volume_to_hit += abs(volume)
                else:
                    break
            qty_to_clear = min(abs(position_after_take), volume_to_hit)
            sent_quantity = min(remaining_buy_capacity, qty_to_clear)
            if sent_quantity > 0:
                orders.append(Order(product, clear_bid_price, sent_quantity))
                buy_order_volume += sent_quantity

        return buy_order_volume, sell_order_volume

    def market_make(
        self,
        product: str,
        orders: List[Order],
        bid: int,
        ask: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        order_depth: OrderDepth,
        volume_limit_factor: float = None
    ) -> (int, int):
        position_limit = self.LIMIT[product]

        # --- Buy Order Calculation ---
        desired_buy_quantity = position_limit - (position + buy_order_volume)
        constrained_buy_quantity = desired_buy_quantity
        if volume_limit_factor is not None and len(order_depth.sell_orders) > 0:
            best_ask_volume = abs(order_depth.sell_orders[min(order_depth.sell_orders.keys())])
            volume_limit = math.floor(best_ask_volume * volume_limit_factor)
            constrained_buy_quantity = min(desired_buy_quantity, volume_limit)
            

        final_buy_quantity = constrained_buy_quantity
        if final_buy_quantity > 0:
         
            orders.append(Order(product, bid, final_buy_quantity))

        # --- Sell Order Calculation ---
        desired_sell_quantity = position_limit + (position - sell_order_volume)
        constrained_sell_quantity = desired_sell_quantity
        if volume_limit_factor is not None and len(order_depth.buy_orders) > 0:
            best_bid_volume = order_depth.buy_orders[max(order_depth.buy_orders.keys())]
            volume_limit = math.floor(best_bid_volume * volume_limit_factor)
            constrained_sell_quantity = min(desired_sell_quantity, volume_limit)
            # if constrained_sell_quantity < desired_sell_quantity:
            #     # logger.print(f"MAKE SELL VOL LIMIT: {product} Desired: {desired_sell_quantity}, BidVol: {best_bid_volume}, LimitFactor: {volume_limit_factor}, Limited: {constrained_sell_quantity}")

        final_sell_quantity = constrained_sell_quantity
        if final_sell_quantity > 0:
            orders.append(Order(product, ask, -final_sell_quantity))

        return buy_order_volume, sell_order_volume

    def take_orders(
        self, product: str, order_depth: OrderDepth, fair_value: float,
        take_width: float, position: int, prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0
        buy_order_volume, sell_order_volume = self.take_best_orders(
            product, fair_value, take_width, orders, order_depth, position,
            buy_order_volume, sell_order_volume, prevent_adverse, adverse_volume,
        )
        logger.print(f"TAKE Orders: {product} Buys: {buy_order_volume}, Sells: {sell_order_volume}")
        return orders, buy_order_volume, sell_order_volume

    def clear_orders(
        self, product: str, order_depth: OrderDepth, fair_value: float,
        clear_width: int, position: int, buy_order_volume: int, sell_order_volume: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume, sell_order_volume = self.clear_position_order(
            product, fair_value, clear_width, orders, order_depth, position,
            buy_order_volume, sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def make_orders(
        self,
        product,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        params: Dict,
        manage_position: bool = False,
    ):
        orders: List[Order] = []
        disregard_edge = params["disregard_edge"]
        join_edge = params["join_edge"]
        default_edge = params["default_edge"]
        soft_position_limit = params.get("soft_position_limit", 0)
        volume_limit_factor = params.get("volume_limit_factor", None)

        asks_outside_disregard = [p for p in order_depth.sell_orders if p > fair_value + disregard_edge]
        bids_outside_disregard = [p for p in order_depth.buy_orders if p < fair_value - disregard_edge]

        best_ask_outside = min(asks_outside_disregard) if asks_outside_disregard else None
        best_bid_outside = max(bids_outside_disregard) if bids_outside_disregard else None

        ask = round(fair_value + default_edge)
        if best_ask_outside is not None:
            if best_ask_outside <= fair_value + join_edge:
                ask = best_ask_outside
            else:
                ask = best_ask_outside - 1
    

        bid = round(fair_value - default_edge)
        if best_bid_outside is not None:
            if best_bid_outside >= fair_value - join_edge:
                bid = best_bid_outside
                logger.print(f"MAKE Bid Join: {product} Fair: {fair_value}, BestBidOut: {best_bid_outside}, JoinEdge: {join_edge}, Bid: {bid}")
            else:
                bid = best_bid_outside + 1
                logger.print(f"MAKE Bid Penny: {product} Fair: {fair_value}, BestBidOut: {best_bid_outside}, Bid: {bid}")
        # else:
        #     # logger.print(f"MAKE Bid Default: {product} Fair: {fair_value}, DefaultEdge: {default_edge}, Bid: {bid}")

        if manage_position and soft_position_limit > 0:
            current_position = position + buy_order_volume - sell_order_volume
            if current_position > soft_position_limit:
                ask = min(ask, bid + 1)
                # logger.print(f"MAKE Pos Adjust Ask: {product} Pos: {current_position}, SoftLimit: {soft_position_limit}, NewAsk: {ask}")
            elif current_position < -soft_position_limit:
                bid = max(bid, ask - 1)
                # logger.print(f"MAKE Pos Adjust Bid: {product} Pos: {current_position}, SoftLimit: {-soft_position_limit}, NewBid: {bid}")

        if bid >= ask:
            
            return orders, buy_order_volume, sell_order_volume

        _, _ = self.market_make(
            product, orders, bid, ask, position,
            buy_order_volume, sell_order_volume,
            order_depth,
            volume_limit_factor
        )

        return orders, buy_order_volume, sell_order_volume
    
    def get_mid_price(self, product, state : TradingState):
        default_price = Product.ema_param
        if default_price is None:
            default_price = self.params[product]["fair_value"]  


        if product not in state.order_depths:
            return default_price

        market_bids = state.order_depths[product].buy_orders
        if len(market_bids) == 0:
           
            return default_price
        
        market_asks = state.order_depths[product].sell_orders
        if len(market_asks) == 0:

            return default_price
        
        best_bid = max(market_bids)
        best_ask = min(market_asks)
        return (best_bid + best_ask)/2
    

    def compute_avg_prices(self, product, state: TradingState):
        order_depth = state.order_depths[product]

        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None

        total_buy_vol = sum(abs(v) for v in order_depth.buy_orders.values())
        avg_buy = sum(abs(v) * p for p, v in order_depth.buy_orders.items()) / total_buy_vol

        # Average sell price (weighted by volume)
        total_sell_vol = sum(abs(v) for v in order_depth.sell_orders.values())
        avg_sell = sum(abs(v) * p for p, v in order_depth.sell_orders.items()) / total_sell_vol

        return (avg_buy + avg_sell) / 2
   
    

    def kelp_strat(self, state: TradingState , product):
        orders = []
        
        position_product = self.get_position(product, state)
        limit = 50

        order_depth = state.order_depths.get(product)
        if not order_depth or not order_depth.buy_orders or not order_depth.sell_orders:
            # logger.print(f"{product} VWAP Strat: Cannot trade, order depth incomplete.")
            return orders

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid_price = (best_bid + best_ask) / 2

        # --- Record mid_price and volume to calculate VWAP ---
        total_buy_vol = sum(abs(v) for v in order_depth.buy_orders.values())
        avg_buy = sum(abs(v) * p for p, v in order_depth.buy_orders.items()) / total_buy_vol

        # Average sell price (weighted by volume)
        total_sell_vol = sum(abs(v) for v in order_depth.sell_orders.values())
        avg_sell = sum(abs(v) * p for p, v in order_depth.sell_orders.items()) / total_sell_vol
        volume_bid = sum(abs(v) for v in order_depth.buy_orders.values())

        volume_ask = sum(abs(v) for v in order_depth.sell_orders.values())
        trade_volume = volume_bid + volume_ask

        avg_wt = self.compute_avg_prices(product,state)


        Product.vwap_history.setdefault(product, []).append((avg_wt, trade_volume))


        if len(Product.vwap_history[product]) > 10:
            Product.vwap_history[product] = Product.vwap_history[product][-10:]

        # Calculate VWAP
        prices, volumes = zip(*Product.vwap_history[product])
        vwap = sum(p * v for p, v in zip(prices, volumes)) / sum(volumes)

        # --- Trading Logic ---
    
        spread = best_ask - best_bid

        ema = Product.ema_prices[product]


        if spread>=1.67:
            if avg_wt < math.ceil(vwap) :
                # Price below VWAP → Buy signal
                volume_to_buy = min(50,limit - position_product)
                orders.append(Order(product,best_bid, volume_to_buy))

            elif avg_wt > math.ceil(vwap):
                # Price above VWAP → Sell signal
                volume_to_sell = min(50,position_product + limit)
                orders.append(Order(product,best_ask, -volume_to_sell))
        else:
            orders.append(Order(product, math.floor(avg_wt) , min(50,50-position_product)))
            orders.append(Order(product, math.ceil(avg_wt), min(-50,-50-position_product)))


        return orders


        
    def squid_s(self,state:TradingState, product):
        position = state.position.get(product,0)

        orders = []

        buy_qty = 50 - position
        sell_qty = -50 - position


        market_bids = state.order_depths[product].buy_orders.keys()
        market_asks = state.order_depths[product].sell_orders.keys()

        
        
        best_bid = max(market_bids)
        best_ask = min(market_asks)

        mid_price = (best_ask + best_bid)/2

        if len(Product.past_prices[product]) < 20 : return orders
        residuals = [p for p in Product.past_prices[product][-20:]]
        # std = np.std(residuals) if len(residuals) >= 2 else 0
        # std = np.std([p for p in self.past_prices[product][-50:]])
        std = np.std(residuals)
        mn = np.mean(residuals)
        logger.print(f"mean : {mn}, std : {std}")

        if std!=0 :

            z = ((mid_price - mn) / std)
            Z_THRESHOLD = 2.54 # Example threshold
            if z >= Z_THRESHOLD: # Price is significantly HIGH (positive Z) -> Sell
                orders.append(Order(product, math.ceil(best_ask - z), sell_qty))
                # logger.print(f"SELL signal: z={z:.2f}, placing order at {best_ask - 1} for {sell_qty}")
            elif z < -Z_THRESHOLD: # Price is significantly LOW (negative Z) -> Buy
                orders.append(Order(product, math.floor(best_bid + z), buy_qty))
                # logger.print(f"BUY signal: z={z:.2f}, placing order at {best_bid + 1} for {buy_qty}")

            else :
                orders.append(Order(product, best_ask  , max(-15,-50-position)))
                orders.append(Order(product, best_bid  , min(15,50-position)))


        
        # if st is None  or mid_price is None:
        return orders

    
    def run_jams_strategy_main(self, state: TradingState) -> List[Order]:
        """
        Implements the Biased Trend & Range strategy for JAMS with:
        - Dynamic Primary Trend Entry Level based on SMA.
        - Trailing Stop-Loss.
        - Partial Profit Taking.
        """
        orders: List[Order] = []
        product = Product.JAMS

        # --- Basic Setup ---
        position = self.get_position(product, state)

        if product not in self.LIMIT:
            logger.print(f"CRITICAL ERROR: Position limit not defined for {product}")
            return []
        limit = self.LIMIT[product]
        position_limit_short = -limit


        # --- Get Market Data ---
        if product not in state.order_depths:
            logger.print(f"JAMS STRAT: No order depth for {product}")
            return orders
        order_depth = state.order_depths[product]
        market_bids = order_depth.buy_orders
        market_asks = order_depth.sell_orders
        if not market_bids or not market_asks:
            
            return orders

        best_bid = max(market_bids.keys())
        best_ask = min(market_asks.keys())
        mid_price = (best_bid + best_ask) / 2

        # --- Dynamic Trend Entry Calculation ---
        jams_prices = Product.past_prices.get(product, [])
        dynamic_primary_trend_entry = self.JAMS_FIXED_RANGE_BOTTOM # Default fallback

        if len(jams_prices) >= self.JAMS_SMA_WINDOW:
            recent_prices = jams_prices[-self.JAMS_SMA_WINDOW:]
            sma = np.mean(recent_prices)
            # Set the dynamic entry level below the SMA
            dynamic_primary_trend_entry = math.floor(sma - self.JAMS_TREND_ENTRY_OFFSET)
            # logger.print(f"JAMS STRAT: Dynamic Entry Calc: SMA({self.JAMS_SMA_WINDOW})={sma:.2f}, Offset={self.JAMS_TREND_ENTRY_OFFSET}, Dynamic Entry Level={dynamic_primary_trend_entry}")
        # else:
        #     logger.print(f"JAMS STRAT: Dynamic Entry Calc: Not enough data ({len(jams_prices)}/{self.JAMS_SMA_WINDOW}), using fixed bottom {self.JAMS_FIXED_RANGE_BOTTOM}")

        # --- Use Calculated/Fixed Levels for Strategy ---
        primary_trend_entry = dynamic_primary_trend_entry # Use the dynamic value
        range_bottom = self.JAMS_FIXED_RANGE_BOTTOM     # Keep range bottom fixed (can be adjusted)
        range_top = self.JAMS_FIXED_RANGE_TOP         # Keep range top fixed

        # Ensure range_bottom is logical relative to trend entry (optional refinement)
        # e.g., range_bottom = max(range_bottom, primary_trend_entry) if MM shouldn't start below trend entry

        # --- State Management for Trailing Stop & Profit Taking ---
        # (Logic remains the same as before)
        if position < 0 and self.previous_jams_position >= 0:
            # logger.print(f"JAMS STRAT: Entered SHORT position. Initializing lowest_low tracking at {mid_price:.2f}")
            self.jams_lowest_low_since_short = mid_price
            self.jams_partial_profit_taken_this_move = False
        elif position >= 0 and self.previous_jams_position < 0:
            # logger.print(f"JAMS STRAT: Exited SHORT position. Resetting lowest_low tracking.")
            self.jams_lowest_low_since_short = float('inf')
            self.jams_partial_profit_taken_this_move = False
        if position < 0:
            self.jams_lowest_low_since_short = min(self.jams_lowest_low_since_short, mid_price)

        # --- Strategy Logic ---

        # I. Trailing Stop-Loss Logic (Only if currently Short)
        if position < 0:
            trailing_stop_price = self.jams_lowest_low_since_short + self.JAMS_TRAILING_STOP_DISTANCE
            # logger.print(f"JAMS STRAT: Short ({position}). Low: {self.jams_lowest_low_since_short:.2f}, Trail Stop: {trailing_stop_price:.2f}") # Verbose logging
            if mid_price >= trailing_stop_price:
                qty_to_buy = abs(position)
                if qty_to_buy > 0:
                    # logger.print(f"JAMS STRAT: TRAILING STOP LOSS triggered at mid {mid_price:.2f} >= {trailing_stop_price:.2f}. Covering {qty_to_buy}.")
                    orders.append(Order(product, best_ask, qty_to_buy))
                    self.previous_jams_position = position
                    return orders # Exit immediately

        # II. Partial Profit Taking Logic (Only if Short, Target Hit, Not Yet Taken)
        if position < 0 and mid_price <= self.JAMS_PARTIAL_PROFIT_TARGET and not self.jams_partial_profit_taken_this_move:
            qty_to_buy_partial = math.floor(abs(position) * self.JAMS_PARTIAL_PROFIT_FRACTION)
            if qty_to_buy_partial > 0:
                # logger.print(f"JAMS STRAT: PARTIAL PROFIT TARGET hit at mid {mid_price:.2f} <= {self.JAMS_PARTIAL_PROFIT_TARGET}. Buying back {qty_to_buy_partial}.")
                orders.append(Order(product, best_ask, qty_to_buy_partial))
                self.jams_partial_profit_taken_this_move = True
                self.previous_jams_position = position
                return orders # Exit after placing partial profit order

        # III. Primary Trend Entry (Aggressive Shorting below DYNAMIC level)
        # (Only if stop/profit didn't trigger)
        if mid_price < primary_trend_entry:
            #  logger.print(f"JAMS STRAT: Price {mid_price:.2f} < {primary_trend_entry} (Dynamic Primary Trend Zone).")
             if position > position_limit_short:
                 qty_to_sell = position_limit_short - position
                 if qty_to_sell < 0:
                      logger.print(f"           Pursuing SHORT limit. Curr: {position}, Target: {position_limit_short}, Sell: {qty_to_sell}")
                      orders.append(Order(product, best_bid, qty_to_sell))
                      self.previous_jams_position = position
                      return orders # Exit after placing aggressive short

        # IV. Range-Bound Market Making [FIXED range_bottom, FIXED range_top)
        # (Only if stop/profit/trend entry didn't trigger)
        elif mid_price >= range_bottom and mid_price < range_top:
            #  logger.print(f"JAMS STRAT: Price {mid_price:.2f} in range [{range_bottom}, {range_top}) (Range MM Zone).")
             # SELL SIDE (Aggressive)
             if position > position_limit_short:
                 qty_to_sell = position_limit_short - position
                 if qty_to_sell < 0:
                     sell_price = best_ask - 1
                    #  logger.print(f"           Placing MM SELL Order: {qty_to_sell} at {sell_price}") # Verbose
                     orders.append(Order(product, sell_price, qty_to_sell))
             # BUY SIDE (Passive)
             if position < limit:
                qty_to_buy = limit - position
                if qty_to_buy > 0:
                    buy_price = best_bid
                    # logger.print(f"           Placing MM BUY Order: {qty_to_buy} at {buy_price}") # Verbose
                    orders.append(Order(product, buy_price, qty_to_buy))

      
        elif mid_price >= range_top:
            
             if position < 0:
                 pass # Holding short, waiting for stop or price drop
                 # logger.print(f"           Holding SHORT position: {position}") # Verbose
             else:
                 # Not short, and above MM range -> Do nothing aggressive
                 # MM logic above might place a passive BUY if not at limit
                 pass
                 # logger.print(f"           Holding FLAT/LONG position. MM Buy possible.") # Verbose

        # --- Update state for next iteration ---
        self.previous_jams_position = position # Update last known position *before* returning

        return orders

    def kelp_similar(self,state:TradingState , product):
        position = state.position.get(product,0)

        orders = []

        buy_qty = 50 - position
        sell_qty = -50 - position


        market_bids = state.order_depths[product].buy_orders.keys()
        market_asks = state.order_depths[product].sell_orders.keys()

        
        
        best_bid = max(market_bids)
        best_ask = min(market_asks)

        mid_price = (best_ask + best_bid)/2

        if len(Product.past_prices[product]) < 22 : return orders
        residuals = [p for p in Product.past_prices[product][-22:]]
        # std = np.std(residuals) if len(residuals) >= 2 else 0
        # std = np.std([p for p in self.past_prices[product][-50:]])
        std = np.std(residuals)
        mn = np.mean(residuals)
        logger.print(f"mean : {mn}, std : {std}")

        avg_wt = self.compute_avg_prices(product , state)

        if std!=0 :

            z = ((mid_price - mn) / std)
            Z_THRESHOLD = 1.67# Example threshold
            if z > Z_THRESHOLD: # Price is significantly HIGH (positive Z) -> Sell
                orders.append(Order(product, math.ceil(avg_wt), sell_qty))
                logger.print(f"SELL signal: z={z:.2f}, placing order at {best_ask - 1} for {sell_qty}")
            elif z < -Z_THRESHOLD: # Price is significantly LOW (negative Z) -> Buy
                orders.append(Order(product, math.floor(avg_wt), buy_qty))
                logger.print(f"BUY signal: z={z:.2f}, placing order at {best_bid + 1} for {buy_qty}")
            else :
                orders.append(Order(product, best_ask  , max(-50,-50-position)))
                orders.append(Order(product, best_bid  , min(50,50-position)))


        
        # if st is None  or mid_price is None:
        return orders
    
    def get_rock_price(self, state: TradingState) -> float:
        """Estimate current price of VOLCANIC_ROCK."""
        order_depth = state.order_depths.get("VOLCANIC_ROCK", None)
        if not order_depth:
            return 10000
        best_bid = max(order_depth.buy_orders.keys(), default=0) if order_depth.buy_orders else 0
        best_ask = min(order_depth.sell_orders.keys(), default=float('inf')) if order_depth.sell_orders else float('inf')
        if best_bid and best_ask != float('inf'):
            return (best_bid + best_ask) / 2
        return best_bid or best_ask or 10000
    
    def volcano_strategy(self, state: TradingState) -> Dict[str, List[Order]]:
        result = {}
        positions = state.position if state.position else {}

        rock_price = self.get_rock_price(state)
        if rock_price > 0:
            self.price_history.append(rock_price)
            if len(self.price_history) > 20:
                self.price_history.pop(0)
        volatility = self.estimate_volatility(self.price_history)

        self.days_to_expiry = max(0, self.days_total - (state.timestamp // 100))
        T = self.days_to_expiry / 252.0

        for product in state.order_depths:
            if not (product == "VOLCANIC_ROCK" or product.startswith("VOLCANIC_ROCK_VOUCHER")):
                continue

            order_depth = state.order_depths[product]
            orders: List[Order] = []
            position = positions.get(product, 0)

            if product == "VOLCANIC_ROCK":
                best_bid = max(order_depth.buy_orders.keys(), default=0)
                best_ask = min(order_depth.sell_orders.keys(), default=float('inf'))

                avg_wt = self.compute_avg_prices("VOLCANIC_ROCK" , state)

                if best_bid and best_ask != float('inf'):
                    mid_price = (best_bid + best_ask) / 2
                    spread = 1
                    max_volume = 10

                    buy_price = int(avg_wt)
                    buy_volume = min(max_volume, self.position_limits[product] - position)
                    if buy_volume > 0 and buy_price > 0:
                        orders.append(Order(product, buy_price, buy_volume))
                        logger.print(f"BUY {product} {buy_volume}x{buy_price}")

                    sell_price = int(avg_wt)
                    sell_volume = min(max_volume, self.position_limits[product] + position)
                    if sell_volume > 0:
                        orders.append(Order(product, sell_price, -sell_volume))
                        logger.print(f"SELL {product} {sell_volume}x{sell_price}")

            if product.startswith("VOLCANIC_ROCK_VOUCHER"):
                strike = self.strike_prices[product]
                fair_value = self.black_scholes_call(
                    S=rock_price, K=strike, T=T, r=self.risk_free_rate, sigma=volatility
                )

                best_bid = max(order_depth.buy_orders.keys(), default=0)
                best_ask = min(order_depth.sell_orders.keys(), default=float('inf'))

                spread = 2
                max_volume = 10

                if best_ask != float('inf') and best_ask < fair_value - spread:
                    buy_volume = min(max_volume, self.position_limits[product] - position)
                    if buy_volume > 0:
                        orders.append(Order(product, best_ask, buy_volume))
                        logger.print(f"BUY {product} {buy_volume}x{best_ask}")

                if best_bid > fair_value + spread:
                    sell_volume = min(max_volume, self.position_limits[product] + position)
                    if sell_volume > 0:
                        orders.append(Order(product, best_bid, -sell_volume))
                        logger.print(f"SELL {product} {sell_volume}x{best_bid}")

                buy_price = int(fair_value - spread)
                buy_volume = min(max_volume, self.position_limits[product] - position)
                if buy_volume > 0 and buy_price > 0:
                    orders.append(Order(product, buy_price, buy_volume))
                    logger.print(f"BUY LIMIT {product} {buy_volume}x{buy_price}")

                sell_price = int(fair_value + spread)
                sell_volume = min(max_volume, self.position_limits[product] + position)
                if sell_volume > 0:
                    orders.append(Order(product, sell_price, -sell_volume))
                    logger.print(f"SELL LIMIT {product} {sell_volume}x{sell_price}")

            result[product] = orders


        trader_data = json.dumps({
            "days_to_expiry": self.days_to_expiry,
            "price_history": self.price_history[-10:],
            "volatility": volatility
        })

        return result
    

    def kelp_similar(self,state:TradingState , product):
        position = state.position.get(product,0)

        orders = []

        buy_qty = 60 - position
        sell_qty = -60 - position


        market_bids = state.order_depths[product].buy_orders.keys()
        market_asks = state.order_depths[product].sell_orders.keys()

        
        
        best_bid = max(market_bids)
        best_ask = min(market_asks)

        mid_price = (best_ask + best_bid)/2

        if len(Product.past_prices[product]) < 27: return orders
        residuals = [p for p in Product.past_prices[product][-27:]]
        std = np.std(residuals)
        mn = np.mean(residuals)
        logger.print(f"mean : {mn}, std : {std}")

        ema = Product.ema_prices[product]
      

        if std!=0 :

            z = ((mid_price - mn) / std)
            Z_THRESHOLD = 1.56 # Example threshold
            z = math.ceil(z)
            if z > Z_THRESHOLD: # Price is significantly HIGH (positive Z) -> Sell
                orders.append(Order(product, best_ask, max(-60,sell_qty)))
            
            elif z < -Z_THRESHOLD: # Price is significantly LOW (negative Z) -> Buy
                orders.append(Order(product, best_bid, min(60,buy_qty)))
               
            else :
                orders.append(Order(product, best_ask  , max(-5,-60-position)))
                orders.append(Order(product,  best_bid , min(5,60-position)))


        
        # if st is None  or mid_price is None:
        return orders
    

    def volcano_strategy(self, state: TradingState) -> Dict[str, List[Order]]:
        result = {}
        positions = state.position if state.position else {}

        rock_price = self.get_rock_price(state)
        if rock_price > 0:
            self.price_history.append(rock_price)
            if len(self.price_history) > 20:
                self.price_history.pop(0)
        volatility = self.estimate_volatility(self.price_history)

        self.days_to_expiry = max(0, self.days_total - (state.timestamp // 100))
        T = self.days_to_expiry / 252.0

        for product in state.order_depths:
            if not (product == "VOLCANIC_ROCK" or product.startswith("VOLCANIC_ROCK_VOUCHER")):
                continue

            order_depth = state.order_depths[product]
            orders: List[Order] = []
            position = positions.get(product, 0)

            if product == "VOLCANIC_ROCK":
                best_bid = max(order_depth.buy_orders.keys(), default=0)
                best_ask = min(order_depth.sell_orders.keys(), default=float('inf'))
                
                if best_bid == 0 or best_ask == float('inf'):
                    result[product] = orders
                    continue  # Skip if we can't form a fair price

                mid_price = (best_bid + best_ask) / 2
                spread = best_ask - best_bid

                # Avoid trading if spread is too wide or price unstable
                # if spread >= 2:
                #     logger.print("Spread too wide, skipping low-risk trade.")
                #     result[product] = orders
                #     continue

                # Estimate local volatility
                
                recent = Product.past_prices[product][-60:]
                std = np.std(recent) if len(recent) >= 2 else 0
                mean_price = np.mean(recent)

                # Avoid trading on spike day (based on std dev z-score)
                z = (mid_price - mean_price) / std if std > 0 else 0
               
                # Risk-free strategy: place small neutral trades around mid_price
                max_volume = 10
                buy_price = int(mean_price)
                sell_price = int(mean_price)

                buy_volume = min(max_volume, self.position_limits[product] - position)
                if buy_volume > 0:
                    orders.append(Order(product, buy_price, buy_volume))
                    logger.print(f"SAFE BUY {product} {buy_volume}x{buy_price}")

                sell_volume = min(max_volume, self.position_limits[product] + position)
                if sell_volume > 0:
                    orders.append(Order(product, sell_price, -sell_volume))
                    logger.print(f"SAFE SELL {product} {sell_volume}x{sell_price}")

                result[product] = orders

            if product.startswith("VOLCANIC_ROCK_VOUCHER"):
                strike = self.strike_prices[product]
                fair_value = self.black_scholes_call(
                    S=rock_price, K=strike, T=T, r=self.risk_free_rate, sigma=volatility
                )

                best_bid = max(order_depth.buy_orders.keys(), default=0)
                best_ask = min(order_depth.sell_orders.keys(), default=float('inf'))

                spread = 1
                max_volume = 15

                if best_ask != float('inf') and best_ask < fair_value - spread:
                    buy_volume = min(max_volume, self.position_limits[product] - position)
                    if buy_volume > 0:
                        orders.append(Order(product, best_ask, buy_volume))
                        logger.print(f"BUY {product} {buy_volume}x{best_ask}")

                if best_bid > fair_value + spread:
                    sell_volume = min(max_volume, self.position_limits[product] + position)
                    if sell_volume > 0:
                        orders.append(Order(product, best_bid, -sell_volume))
                        logger.print(f"SELL {product} {sell_volume}x{best_bid}")

                buy_price = int(fair_value - spread)
                buy_volume = min(max_volume, self.position_limits[product] - position)
                if buy_volume > 0 and buy_price > 0:
                    orders.append(Order(product, buy_price, buy_volume))
                    logger.print(f"BUY LIMIT {product} {buy_volume}x{buy_price}")

                sell_price = int(fair_value + spread)
                sell_volume = min(max_volume, self.position_limits[product] + position)
                if sell_volume > 0:
                    orders.append(Order(product, sell_price, -sell_volume))
                    logger.print(f"SELL LIMIT {product} {sell_volume}x{sell_price}")

            result[product] = orders


        trader_data = json.dumps({
            "days_to_expiry": self.days_to_expiry,
            "price_history": self.price_history[-10:],
            "volatility": volatility
        })

        return result
    

    def MAGNIFICENT_MACARONS_implied_bid_ask(self, observation):
        # Check if attributes exist and are not None before calculation
        bid_price = getattr(observation, 'bidPrice', None)
        ask_price = getattr(observation, 'askPrice', None)
        export_tariff = getattr(observation, 'exportTariff', 0) # Default to 0 if missing
        import_tariff = getattr(observation, 'importTariff', 0) # Default to 0 if missing
        transport_fees = getattr(observation, 'transportFees', 0) # Default to 0 if missing

        if bid_price is None or ask_price is None:
             logger.print("Warning: Missing bidPrice or askPrice in conversion observation.")
             return None, None # Cannot calculate if core prices are missing

        # Calculate implied prices
        implied_bid = bid_price - export_tariff - transport_fees - 0.1 # Small adjustment factor
        implied_ask = ask_price + import_tariff + transport_fees
        return implied_bid, implied_ask


    def MAGNIFICENT_MACARONS_adap_edge(self, timestamp: int, position: int):
        # Access params specific to this product
        prod_params = self.params[Product.MAGNIFICENT_MACARONS]

        # Initialize state for this product in traderObject if not present
        if Product.MAGNIFICENT_MACARONS not in self.traderObject:
            self.traderObject[Product.MAGNIFICENT_MACARONS] = {
                "curr_edge": prod_params["init_make_edge"],
                "volume_history": [],
                "optimized": False
            }
        
        # Get current state for this product
        product_state = self.traderObject[Product.MAGNIFICENT_MACARONS]
        curr_edge = product_state["curr_edge"]

        # Handle timestamp 0 initialization
        if timestamp == 0:
            product_state["curr_edge"] = prod_params["init_make_edge"]
            return product_state["curr_edge"]

        # Update volume history
        product_state["volume_history"].append(abs(position))
        if len(product_state["volume_history"]) > prod_params["volume_avg_timestamp"]:
            product_state["volume_history"].pop(0)

        # Check if enough history exists and optimization is not yet done
        if len(product_state["volume_history"]) >= prod_params["volume_avg_timestamp"] and not product_state["optimized"]:
            volume_avg = np.mean(product_state["volume_history"])
            
            # Increase edge if average volume is too high
            if volume_avg >= prod_params["volume_bar"]:
                product_state["volume_history"] = [] # Reset history after adjustment
                product_state["curr_edge"] = curr_edge + prod_params["step_size"]
                # No need to set optimized = True here, allow further increases
            
            # Decrease edge if volume is low and conditions met
            elif (prod_params["dec_edge_discount"] * prod_params["volume_bar"] * (curr_edge - prod_params["step_size"])) > (volume_avg * curr_edge):
                 new_edge = curr_edge - prod_params["step_size"]
                 if new_edge > prod_params["min_edge"]:
                     product_state["volume_history"] = [] # Reset history
                     product_state["curr_edge"] = new_edge
                     product_state["optimized"] = True # Mark as optimized after decreasing edge
                 else:
                     # If reducing goes below min_edge, set to min_edge
                     product_state["curr_edge"] = prod_params["min_edge"]
                     product_state["optimized"] = True # Mark as optimized (at min edge)

        # Return the current edge (which might have been updated)
        return product_state["curr_edge"]


    def MAGNIFICENT_MACARONS_arb_take(self, order_depth: OrderDepth, observation, adap_edge: float, position: int) :
        orders = []
        position_limit = self.LIMIT[Product.MAGNIFICENT_MACARONS]
        conversion_limit = 10 # Define conversion limit
        buy_order_volume = 0
        sell_order_volume = 0

        implied_bid, implied_ask = self.MAGNIFICENT_MACARONS_implied_bid_ask(observation)
        if implied_bid is None or implied_ask is None:
             return [], 0, 0 # Cannot proceed without implied prices

        # Use adaptive edge for taking liquidity
        edge = max(adap_edge, self.params[Product.MAGNIFICENT_MACARONS]["min_edge"]) # Ensure edge is at least min_edge

        # Calculate remaining capacity respecting position limit and conversion limit
        buy_capacity = position_limit - position
        sell_capacity = position_limit + position

        # Take sell orders (buy from market) if price is below adjusted implied bid
        for price in sorted(order_depth.sell_orders.keys()):
            if price < implied_bid - edge:
                available_volume = abs(order_depth.sell_orders[price])
                # Can only buy up to remaining capacity and conversion limit
                potential_buy_volume = min(buy_capacity - buy_order_volume, conversion_limit - buy_order_volume)
                volume_to_take = min(available_volume, potential_buy_volume)

                if volume_to_take > 0:
                    orders.append(Order(Product.MAGNIFICENT_MACARONS, round_price(price), volume_to_take))
                    buy_order_volume += volume_to_take
                    logger.print(f"Arb Take: Buying {volume_to_take} at {round_price(price)}")
                    if buy_order_volume >= conversion_limit or buy_order_volume >= buy_capacity:
                        break # Stop if conversion limit reached or position limit reached for buys
            else:
                 # Prices are sorted, no need to check further
                 break


        # Take buy orders (sell to market) if price is above adjusted implied ask
        for price in sorted(order_depth.buy_orders.keys(), reverse=True):
            if price > implied_ask + edge:
                available_volume = abs(order_depth.buy_orders[price])
                 # Can only sell up to remaining capacity and conversion limit
                potential_sell_volume = min(sell_capacity - sell_order_volume, conversion_limit - sell_order_volume)
                volume_to_take = min(available_volume, potential_sell_volume)

                if volume_to_take > 0:
                    orders.append(Order(Product.MAGNIFICENT_MACARONS, round_price(price), -volume_to_take))
                    sell_order_volume += volume_to_take
                    logger.print(f"Arb Take: Selling {volume_to_take} at {round_price(price)}")
                    if sell_order_volume >= conversion_limit or sell_order_volume >= sell_capacity:
                        break # Stop if conversion limit reached or position limit reached for sells
            else:
                 # Prices are sorted reverse, no need to check further
                 break

        return orders, buy_order_volume, sell_order_volume


    def MAGNIFICENT_MACARONS_arb_make(self, order_depth: OrderDepth, observation, position: int, adap_edge: float, buy_order_volume: int, sell_order_volume: int) :
        orders = []
        position_limit = self.LIMIT[Product.MAGNIFICENT_MACARONS]
        conversion_limit = 10 # Define conversion limit
        prod_params = self.params[Product.MAGNIFICENT_MACARONS]

        implied_bid, implied_ask = self.MAGNIFICENT_MACARONS_implied_bid_ask(observation)
        if implied_bid is None or implied_ask is None:
            return [], 0, 0 # Cannot proceed without implied prices

        # Use adaptive edge for making orders, ensuring it's not below make_min_edge
        edge = max(adap_edge, prod_params["make_min_edge"])

        # Calculate base bid and ask prices for making orders
        bid_price = round_price(implied_bid - edge)
        ask_price = round_price(implied_ask + edge)

        # Calculate remaining capacity, considering orders already placed by arb_take
        # Max volume per side for making orders is conversion_limit minus volume already taken
        remaining_buy_make_capacity = max(0, conversion_limit - buy_order_volume)
        remaining_sell_make_capacity = max(0, conversion_limit - sell_order_volume)

        # Calculate position capacity
        position_buy_capacity = position_limit - (position + buy_order_volume)
        position_sell_capacity = position_limit + (position - sell_order_volume) # Note: position is negative for short

        # Place buy order (make bid)
        buy_quantity = min(remaining_buy_make_capacity, position_buy_capacity)
        if buy_quantity > 0:
            orders.append(Order(Product.MAGNIFICENT_MACARONS, bid_price, buy_quantity))
            logger.print(f"Arb Make: Bidding {buy_quantity} at {bid_price}")


        # Place sell order (make ask)
        # Sell quantity is negative
        sell_quantity = min(remaining_sell_make_capacity, position_sell_capacity)
        if sell_quantity > 0:
            orders.append(Order(Product.MAGNIFICENT_MACARONS, ask_price, -sell_quantity))
            logger.print(f"Arb Make: Asking {sell_quantity} at {ask_price}")

        # Return make orders (buy_order_volume, sell_order_volume are not updated here, they reflect 'take' volume)
        return orders, buy_order_volume, sell_order_volume


    # --- Main Strategy Function ---
    def MAGNIFICENT_MACARONS_strategy(self, state: TradingState):
        """
        Decides trading strategy based on sunlight index compared to CSI.
        - Below CSI: Aggressive directional buying.
        - Above or equal CSI: Arbitrage around conversion prices.
        """
         # Basic checks for necessary data
        if (Product.MAGNIFICENT_MACARONS not in state.order_depths or
            state.observations is None or
            Product.MAGNIFICENT_MACARONS not in state.observations.conversionObservations):
             logger.print("Warning: Missing critical data for MAGNIFICENT_MACARONS.")
             return [], 0 # Return empty orders and 0 conversions

        order_depth: OrderDepth = state.order_depths[Product.MAGNIFICENT_MACARONS]
        observation = state.observations.conversionObservations[Product.MAGNIFICENT_MACARONS]
        position = self.get_position(Product.MAGNIFICENT_MACARONS, state)
        position_limit = self.LIMIT[Product.MAGNIFICENT_MACARONS]
        prod_params = self.params[Product.MAGNIFICENT_MACARONS]
        csi = prod_params["csi"]

        # Get sunlightIndex safely
        sunlight_index = getattr(observation, 'sunlightIndex', None)
        if sunlight_index is None:
            logger.print("Warning: sunlightIndex not found in observation. Defaulting to Normal Regime.")
            sunlight_index = csi + 1 # Default to normal regime if data missing

        orders = []
        conversions = 0 # Initialize conversions for this round

        # --- Regime Switching Logic ---
        if sunlight_index < csi:
            # --- Low Sunlight Regime ---
            logger.print(f"MAGNIFICENT_MACARONS: Low Sunlight Regime (Sunlight={sunlight_index} < CSI={csi})")

            # Strategy: Aggressively buy towards the positive position limit
            buy_capacity = position_limit - position
            if buy_capacity > 0:
                # Determine aggressive buy price: target slightly above best ask
                best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
                buy_price = None

                if best_ask:
                    buy_price = round_price(best_ask + prod_params["low_sun_aggr_price_offset"])
                else:
                    # If no asks, use implied ask from conversion + offset as fallback
                     _, implied_ask = self.MAGNIFICENT_MACARONS_implied_bid_ask(observation)
                     if implied_ask:
                         buy_price = round_price(implied_ask + prod_params["low_sun_aggr_price_offset"] + 1) # Extra offset if using implied

                # Determine order volume: Use specific param, limit by capacity
                order_volume = min(buy_capacity, prod_params["low_sun_aggr_buy_vol"])

                if buy_price is not None and order_volume > 0:
                    logger.print(f"Low Sunlight: Aggressive Buy Order -> {order_volume} units at {buy_price}")
                    orders.append(Order(Product.MAGNIFICENT_MACARONS, buy_price, order_volume))
                else:
                    logger.print(f"Low Sunlight: Cannot place buy order (capacity={buy_capacity}, price={buy_price}, volume={order_volume})")

            # In this regime, avoid selling unless absolutely necessary (e.g., over limit - should not happen with check above)
            # No active conversions planned in this regime, focus on market position

        else:
            # --- Normal Sunlight Regime (Sunlight >= CSI) ---
            logger.print(f"MAGNIFICENT_MACARONS: Normal Sunlight Regime (Sunlight={sunlight_index} >= CSI={csi})")

            # Use the existing arbitrage logic based on conversion prices
            # Calculate adaptive edge based on position history
            adap_edge = self.MAGNIFICENT_MACARONS_adap_edge(state.timestamp, position)
            logger.print(f"Normal Sunlight: Adaptive Edge = {adap_edge:.2f}")


            # 1. Try to take profitable arbitrage opportunities
            take_orders, buy_volume_taken, sell_volume_taken = self.MAGNIFICENT_MACARONS_arb_take(
                order_depth, observation, adap_edge, position
            )
            orders.extend(take_orders)

            # 2. Place making orders based on remaining capacity and edge
            # Pass the volume already taken to arb_make to respect conversion limits per tick
            make_orders, _, _ = self.MAGNIFICENT_MACARONS_arb_make(
                order_depth, observation, position, adap_edge, buy_volume_taken, sell_volume_taken
            )
            orders.extend(make_orders)

            # Conversion logic could be added here for position management if needed,
            # but the arb logic already implicitly uses conversion prices.
            # Keeping conversions = 0 for now.

        return orders, conversions # Return calculated orders and conversions for this product

    



    
    def run(self, state: TradingState):
        self.round += 1
        pnl = self.update_pnl(state)
        self.update_ema_prices(state)
      
        logger.print(f"Round {self.round}")
        logger.print(f"\tCash {self.cash}")
        for p in [Product.RAINFOREST_RESIN, Product.KELP, Product.SQUID_INK]:
            logger.print(f"\t{p} Pos: {self.get_position(p, state)}, Mid: {self.get_mid_price(p, state)}, Value: {self.get_value_on_product(p, state)}, EMA: {Product.ema_prices[p]}")
        logger.print(f"\tPnL {pnl}")

        traderObject = {}
        if state.traderData is not None and state.traderData != "":
            try:
                traderObject = jsonpickle.decode(state.traderData)
            except Exception as e:
                logger.print(f"Error decoding traderData: {e}")
                traderObject = {}

        self.update_ema_prices(state)
        self.update_past_prices(state)

        result = {}
        product1 = Product.RAINFOREST_RESIN
        # try:
        #     result[product1] = self.resin_strategy(state)
        # except Exception as e:
        #     logger.print("Error in PRODUCT1:", e)

 
        # product2 = Product.KELP

        # try:
        #     result[product2] = self.market_make_kelp(state)
        # except Exception as e:
        #     logger.print("Error in PRODUCT1:", e)

        # product3 = Product.SQUID_INK

        # try:
        #     result[product3] = self.squid_s(state,product3)
        # except Exception as e:
        #     logger.print("Error in PRODUCT1:", e)

        # product5 = Product.DJEMBES

        # try:
        #     result[product5] = self.kelp_similar(state,product5)
        # except Exception as e:
        #     logger.print("Error in PRODUCT1:", e)


        product4 = Product.JAMS
       

        product6 = Product.CROISSANTS
        

        try:
            result[product6],result[product4] = self.pair_strategy(state)
        except Exception as e:
            logger.print("Error in pair strategy")
            logger.print(e)
            

        basket_position = (
            state.position[Product.BASKET1]
            if Product.BASKET1 in state.position
            else 0
        )
        basket_position2 = (
            state.position[Product.BASKET2]
            if Product.BASKET2 in state.position
            else 0
        )

        spread_orders = self.spread_orders(
            state.order_depths,
            Product.BASKET1,
            basket_position,
            self.Spread
        )
        if spread_orders != None:
            result[Product.BASKET1] = spread_orders[Product.BASKET1]

        spread_orders_2 = self.spread_orders_2(
            state.order_depths,
            Product.BASKET2,
            basket_position2,
            self.Spread2
        )
        if spread_orders_2 != None:
            result[Product.BASKET2] = spread_orders_2[Product.BASKET2]


        # try:
        #     volcano_result = self.volcano_strategy(state)
        #     result.update(volcano_result)
        # except Exception as e:
        #     logger.print("Error in volcano strategy:", e)


       
        if state.traderData is not None and state.traderData != "":
         
            # Use safe=True if there might be untrusted data
            decoded_data = jsonpickle.decode(state.traderData)
            if isinstance(decoded_data, dict): # Ensure it decodes to a dictionary
                self.traderObject = decoded_data
    


        # Initialize result
        conversions = 0

        # # --- Execute Strategy for MAGNIFICENT_MACARONS ---
        # product = Product.MAGNIFICENT_MACARONS
        # if product in state.listings: # Check if product is traded in this round/day
        #     try:
        #         # Call the specific strategy function for the product
        #         orders, t_conversions = self.MAGNIFICENT_MACARONS_strategy(state)
        #         result[product] = orders
        #         conversions += t_conversions # Accumulate conversions

        #     except Exception as e:
        #         logger.print(f"Error occurred in {product} strategy: {e}")
        #         result[product] = [] # Return empty list on error to avoid crash

        # else:
        #      logger.print(f"{product} not in listings for this round.")


        traderData = jsonpickle.encode(self.traderObject, unpicklable=False)



        logger.flush(state, result, conversions, traderData)

        return result, conversions, traderData