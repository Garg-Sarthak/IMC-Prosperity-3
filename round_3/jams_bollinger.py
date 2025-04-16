from typing import Dict, List
import numpy as np
import json
import pandas as pd
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
RESIN = "RAINFOREST_RESIN"
KELP = "KELP" 
CROISSANTS = "CROISSANTS"
JAMS = "JAMS"


KELP = "KELP"
RESIN = "RAINFOREST_RESIN"
SQUID = "SQUID_INK"
DJEMBES = "DJEMBES"
PRODUCTS = [
    CROISSANTS,
    JAMS,
    DJEMBES
]

DEFAULT_PRICES = {
    RESIN : 10000,
    KELP : 2016,
    SQUID : 2040,
    CROISSANTS : 4298,
    JAMS : 6593,
    DJEMBES : 13435
}

POSITION_LIMITS = {
    CROISSANTS: 250,
    JAMS: 350,
    DJEMBES : 60
}

ORDER_VOLUME = 5
WINDOW = 150


class Trader:

    def __init__(self) -> None:
        
        logger.print("Initializing Trader...")

        self.position_limit = {
            RESIN : 50,
            KELP : 50,
            SQUID : 50,
            CROISSANTS: 250,
            JAMS: 350,
            DJEMBES : 60
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

        self.ema_param = 0.01

        self.all_positions = set()

        self.crossover_state = -1
        self.crossover_state_2 = -1
        
        self.prices = {
            JAMS: pd.Series(dtype=float),
            CROISSANTS: pd.Series(dtype=float),
            DJEMBES : pd.Series(dtype=float),
            "Spread": pd.Series(dtype=float),
            "Spread2": pd.Series(dtype=float)
        }

        self.all_positions = set()
        self.coconuts_pair_position = 0
        self.croissants_pair_position = 0


    # utils

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

 
    def save_prices(self, state: TradingState):
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
        spread_value = price_jams - 1.52990725*price_croissants
        self.prices["Spread"] = pd.concat([
            self.prices["Spread"],
            pd.Series({timestamp: spread_value}, dtype=float)
        ])
    
    def save_prices_2(self, state: TradingState):
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

    def croissant_strategy(self,state: TradingState):
        orders_croissants = []
        self.save_prices(state)

        short_len = 5
        long_len = 9

        if len(self.prices[CROISSANTS]) < long_len:
            return orders_croissants
        
        mid_price_croissants = self.get_mid_price(CROISSANTS, state)
        int_price_croissants = int(mid_price_croissants)
        croissants_position = self.get_position(CROISSANTS, state)


        short_avg = self.prices[CROISSANTS].rolling(short_len).mean().iloc[-1]
        long_avg = self.prices[CROISSANTS].rolling(long_len).mean().iloc[-1]

        prevState = self.crossover_state
        self.crossover_state = 1 if short_avg > long_avg else 0

        if prevState != self.crossover_state:
            if prevState == 1 and self.crossover_state == 0: # go short
                order_quantity = -POSITION_LIMITS[CROISSANTS] - croissants_position
                orders_croissants.append(Order(CROISSANTS, int_price_croissants, order_quantity))
            elif prevState == 0 and self.crossover_state == 1: # go long
                order_quantity = POSITION_LIMITS[CROISSANTS] - croissants_position
                orders_croissants.append(Order(CROISSANTS, int_price_croissants, order_quantity))

        return orders_croissants

    def croissants_strategy_3(self, state: TradingState):
        orders_croissants = []
        position_croissants = self.get_position(CROISSANTS, state)

        self.save_prices(state)

        #best 11058 at 200,1.75,4
        window = 200

        if len(self.prices[CROISSANTS]) < window:
            return orders_croissants
        
        mid_price_croissants = self.get_mid_price(CROISSANTS, state)
        int_price_croissants = int(mid_price_croissants)

        mean_croissants = self.prices[CROISSANTS].rolling(window).mean().iloc[-1]
        std_croissants = self.prices[CROISSANTS].rolling(window).std().iloc[-1]

        threshold = 1.75

        z_score = (mid_price_croissants - mean_croissants) / std_croissants
        signal_strength = abs(z_score)/3

        if z_score > threshold:
            order_quantity = POSITION_LIMITS[CROISSANTS] - position_croissants
            order_quantity = int(order_quantity * signal_strength)
            orders_croissants.append(Order(CROISSANTS, int_price_croissants, order_quantity))
        elif z_score < -threshold:
            order_quantity = -POSITION_LIMITS[CROISSANTS] - position_croissants
            order_quantity = int(order_quantity * signal_strength)
            orders_croissants.append(Order(CROISSANTS, int_price_croissants, order_quantity))

        return orders_croissants

    def jams_strategy_3(self, state: TradingState):
        orders_jams = []
        position_jams = self.get_position(JAMS, state)

        self.save_prices(state)

        window = 150

        if len(self.prices[JAMS]) < window:
            return orders_jams
        
        mid_price_jams = self.get_mid_price(JAMS, state)
        int_price_jams = int(mid_price_jams)

        mean_jams = self.prices[JAMS].rolling(window).mean().iloc[-1]
        std_jams = self.prices[JAMS].rolling(window).std().iloc[-1]

        threshold = 1.0

        z_score = (mid_price_jams - mean_jams) / std_jams
        signal_strength = abs(z_score)/2  # Maintain original signal scaling

        # Position management logic
        if z_score > threshold:
            order_quantity = POSITION_LIMITS[JAMS] - position_jams
            order_quantity = int(order_quantity * signal_strength)
            orders_jams.append(Order(JAMS, int_price_jams, order_quantity))
        elif z_score < -threshold:
            order_quantity = -POSITION_LIMITS[JAMS] - position_jams
            order_quantity = int(order_quantity * signal_strength)
            orders_jams.append(Order(JAMS, int_price_jams, order_quantity))

        return orders_jams


    def djembes_strategy(self,state: TradingState):
        # orders_croissants = []
        orders_djembes = []
        self.save_prices_2(state)

        short_len = 9
        long_len = 14

        if len(self.prices[DJEMBES]) < long_len:
            return orders_djembes
        
        mid_price_djembes = self.get_mid_price(DJEMBES, state)
        int_price_djembes = int(mid_price_djembes)
        djembes_position = self.get_position(DJEMBES, state)


        short_avg = self.prices[DJEMBES].rolling(short_len).mean().iloc[-1]
        long_avg = self.prices[DJEMBES].rolling(long_len).mean().iloc[-1]

        prevState = self.crossover_state_2
        self.crossover_state_2 = 1 if short_avg > long_avg else 0

        if prevState != self.crossover_state_2:
            if prevState == 1 and self.crossover_state_2 == 0: # go short
                order_quantity = -POSITION_LIMITS[DJEMBES] - djembes_position
                orders_djembes.append(Order(DJEMBES, int_price_djembes, order_quantity))
            elif prevState == 0 and self.crossover_state_2 == 1: # go long
                order_quantity = POSITION_LIMITS[DJEMBES] - djembes_position
                orders_djembes.append(Order(DJEMBES, int_price_djembes, order_quantity))

        return orders_djembes

    def djembes_strategy_2(self,state:TradingState):
        orders_djembes = []
        position_djembes = self.get_position(DJEMBES, state)

        bid_volume = self.position_limit[DJEMBES] - position_djembes
        ask_volume = - self.position_limit[DJEMBES] - position_djembes

        if position_djembes == 0:
            orders_djembes.append(Order(DJEMBES, math.floor(self.ema_prices[DJEMBES] - 4), bid_volume))
            orders_djembes.append(Order(DJEMBES, math.ceil(self.ema_prices[DJEMBES] + 4), ask_volume))
        
        if position_djembes > 0:
            orders_djembes.append(Order(DJEMBES, math.floor(self.ema_prices[DJEMBES] - 6), bid_volume))
            orders_djembes.append(Order(DJEMBES, math.ceil(self.ema_prices[DJEMBES]), ask_volume))

        if position_djembes < 0:    
            orders_djembes.append(Order(DJEMBES, math.floor(self.ema_prices[DJEMBES]), bid_volume))
            orders_djembes.append(Order(DJEMBES, math.ceil(self.ema_prices[DJEMBES] + 6), ask_volume))

        return orders_djembes

    def djembes_strategy_3(self,state:TradingState):
        orders_djembes = []
        position_djembes = self.get_position(DJEMBES, state)

        self.save_prices_2(state)

        #best 11058 at 200,1.75,4

        window = 200

        if len(self.prices[DJEMBES]) < window:
            return orders_djembes
        
        mid_price_djembes = self.get_mid_price(DJEMBES, state)
        int_price_djembes = int(mid_price_djembes)

        mean_djembes = self.prices[DJEMBES].rolling(window).mean().iloc[-1]
        std_djembes = self.prices[DJEMBES].rolling(window).std().iloc[-1]

        threshold = 1.75

        z_score = (mid_price_djembes - mean_djembes) / std_djembes
        signal_strength = int(abs(z_score)//1.345)

        if z_score > threshold:
            order_quantity = POSITION_LIMITS[DJEMBES] - position_djembes
            order_quantity = int(order_quantity * signal_strength)
            orders_djembes.append(Order(DJEMBES, int_price_djembes, order_quantity))
        elif z_score < -threshold:
            order_quantity = -POSITION_LIMITS[DJEMBES] - position_djembes
            order_quantity = int(order_quantity * signal_strength)
            orders_djembes.append(Order(DJEMBES, int_price_djembes, order_quantity))

        return orders_djembes


    def pair_strategy(self, state: TradingState):
        orders_croissants = []
        orders_jams = []

        self.save_prices(state)

        if len(self.prices["Spread"]) < WINDOW:
            logger.print(f"Not enough data for window {WINDOW}, have {len(self.prices['Spread'])}")
            return orders_croissants, orders_jams

        mid_price_croissants = self.get_mid_price(CROISSANTS, state)
        mid_price_jams = self.get_mid_price(JAMS, state)

        int_price_croissants = int(mid_price_croissants)
        int_price_jams = int(mid_price_jams)

        croissants_position = self.get_position(CROISSANTS, state)
        jams_position = self.get_position(JAMS, state)

        avg_spread = self.prices["Spread"].rolling(WINDOW).mean()
        std_spread = self.prices["Spread"].rolling(WINDOW).std()
        spread_short = self.prices["Spread"].rolling(5).mean()

        avg_spread = avg_spread.iloc[-1]
        std_spread = std_spread.iloc[-1]
        spread_short = spread_short.iloc[-1]

        logger.print(spread_short)

        # logger.print(f"Spread: {spread_short}, avg: {avg_spread}, std: {std_spread}")

        # - croissants -coconuts
        price_adj = 0
        threshold = 1.0
        ORDER_VOLUME = 10

        if abs(croissants_position) < POSITION_LIMITS[CROISSANTS] - 10:
            if spread_short < avg_spread - threshold*std_spread:
                orders_croissants.append(Order(CROISSANTS, int_price_croissants-price_adj, -ORDER_VOLUME))
                orders_jams.append(Order(JAMS, int_price_jams+price_adj, (int)(ORDER_VOLUME*1.4)))
                self.croissants_pair_position -= ORDER_VOLUME

            elif spread_short > avg_spread + threshold*std_spread:
                orders_croissants.append(Order(CROISSANTS, int_price_croissants+price_adj, ORDER_VOLUME))
                orders_jams.append(Order(JAMS, int_price_jams-price_adj, (int)(-ORDER_VOLUME*1.4)))
                self.croissants_pair_position += ORDER_VOLUME
        else :
            if croissants_position > 0:
                if spread_short < avg_spread - threshold*std_spread:
                    orders_croissants.append(Order(CROISSANTS, int_price_croissants-price_adj, -ORDER_VOLUME))
                    orders_jams.append(Order(JAMS, int_price_jams+price_adj, (int)(ORDER_VOLUME*1.4)))
                    self.croissants_pair_position -= ORDER_VOLUME
            else :
                if spread_short > avg_spread + threshold*std_spread:
                    orders_croissants.append(Order(CROISSANTS, int_price_croissants+price_adj, ORDER_VOLUME))
                    orders_jams.append(Order(JAMS, int_price_jams-price_adj, (int)(-ORDER_VOLUME*1.4)))
                    self.croissants_pair_position += ORDER_VOLUME

        return orders_croissants, orders_jams

    def pair_strategy_2(self, state: TradingState):
        orders_croissants = []
        orders_jams = []

        self.save_prices_2(state)

        if len(self.prices["Spread2"]) < WINDOW:
            logger.print(f"Not enough data for window {WINDOW}, have {len(self.prices['Spread2'])}")
            return orders_croissants, orders_jams

        mid_price_croissants = self.get_mid_price(CROISSANTS, state)
        mid_price_jams = self.get_mid_price(DJEMBES, state)

        int_price_croissants = int(mid_price_croissants)
        int_price_jams = int(mid_price_jams)

        # croissants_position = self.croissants_pair_position
        croissants_position = self.get_position(CROISSANTS, state)
        jams_position = self.get_position(DJEMBES, state)

        avg_spread = self.prices["Spread2"].rolling(WINDOW).mean()
        std_spread = self.prices["Spread2"].rolling(WINDOW).std()
        spread_short = self.prices["Spread2"].rolling(5).mean()

        avg_spread = avg_spread.iloc[-1]
        std_spread = std_spread.iloc[-1]
        spread_short = spread_short.iloc[-1]

        logger.print(spread_short,avg_spread, std_spread)

        # logger.print(f"Spread: {spread_short}, avg: {avg_spread}, std: {std_spread}")

        # - croissants -coconuts
        price_adj = 0
        threshold = 1.25

        ORDER_VOLUME = 16
        if abs(croissants_position) < POSITION_LIMITS[CROISSANTS] - 250%ORDER_VOLUME:
        # if croissants_position <= 0:
            if spread_short < avg_spread - threshold*std_spread:
                orders_croissants.append(Order(CROISSANTS, int_price_croissants-price_adj, -ORDER_VOLUME))
                orders_jams.append(Order(DJEMBES, int_price_jams+price_adj, int(ORDER_VOLUME//4.16667)))

            elif spread_short > avg_spread + threshold*std_spread:
                orders_croissants.append(Order(CROISSANTS, int_price_croissants+price_adj, ORDER_VOLUME))
                orders_jams.append(Order(DJEMBES, int_price_jams-price_adj, int(-ORDER_VOLUME//4.16667)))
        else :
            if croissants_position > 0:
                if spread_short < avg_spread - threshold*std_spread:
                    orders_croissants.append(Order(CROISSANTS, int_price_croissants-price_adj, -ORDER_VOLUME))
                    orders_jams.append(Order(DJEMBES, int_price_jams+price_adj, int(ORDER_VOLUME//4.16667)))
            else :
                if spread_short > avg_spread + threshold*std_spread:
                    orders_croissants.append(Order(CROISSANTS, int_price_croissants+price_adj, ORDER_VOLUME))
                    orders_jams.append(Order(DJEMBES, int_price_jams-price_adj, int(-ORDER_VOLUME//4.16667)))

        return orders_croissants, orders_jams
        
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
        
        result = {}
        try:
            # result[CROISSANTS],result[JAMS] = self.pair_strategy(state)
            # result[CROISSANTS],result[DJEMBES] = self.pair_strategy_2(state)
            # result[CROISSANTS] = self.croissant_strategy(state)
            # result[CROISSANTS] = self.croissants_strategy_3(state)
            # result[DJEMBES] = self.djembes_strategy(state)
            # result[DJEMBES] = self.djembes_strategy_2(state)
            # result[DJEMBES] = self.djembes_strategy_3(state)
            result[JAMS] = self.jams_strategy_3(state)
        except Exception as e:
            logger.print("Error in pair strategy")
            logger.print(e)

        logger.print("+---------------------------------+")
        logger.flush(state, result, conversions=0, trader_data="SAMPLE")

        return result, 0, "SAMPLE"