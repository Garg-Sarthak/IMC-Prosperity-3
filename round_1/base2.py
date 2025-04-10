from typing import Dict, List, Union
import numpy as np
import pandas as pd
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
        
        print("Initializing Trader... ok")

        self.position_limit = {
            PRODUCT1 : 50,
            PRODUCT2 : 50,
            "COCONUTS_EMA": 300
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

        # self.prices : Dict[PRODUCTS, pd.Series] = {
        #     PINA_COLADAS: pd.Series(),
        #     COCONUTS: pd.Series(),
        #     "Spread":pd.Series(),
        #     DIVING_GEAR:pd.Series(),
        #     "SPREAD_PICNIC": pd.Series(),
        # }

        self.all_positions = set()

        self.coconuts_pair_position = 0
        self.last_dolphin_price = -1
        self.dolphin_signal = 0 # 0 if closed, 1 long, -1 short
        self.trend = 0

        self.min_time_hold_position = 20 * 100
        self.initial_time_hold_position = 0

        # Olivia
        self.olivia_buy_trend = False
        self.memory_olivia = False

    # utils
    # gets curr position in product
    def get_position(self, product, state : TradingState):
        return state.position.get(product, 0)    

    #get mid price from order depths = (best ask + best bid) / 2
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

    #current cash in product
    def get_value_on_product(self, product, state : TradingState):
        """
        Returns the amount of MONEY currently held on the product.  
        """
        return self.get_position(product, state) * self.get_mid_price(product, state)
            
    #seems useless
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

    
    # def save_prices(self, state: TradingState):
    #     price_coconut = self.get_mid_price(COCONUTS, state)
    #     price_pina_colada = self.get_mid_price(PINA_COLADAS, state)

    #     self.prices[COCONUTS] = pd.concat([
    #         self.prices[COCONUTS], 
    #         pd.Series({state.timestamp: price_coconut})
    #     ])

    #     self.prices[PINA_COLADAS] = pd.concat([
    #         self.prices[PINA_COLADAS],
    #         pd.Series({state.timestamp: price_pina_colada})
    #     ])

    #     self.prices["Spread"] = self.prices[PINA_COLADAS] - 1.551*self.prices[COCONUTS]

    def save_prices_product(
            self, 
            product, 
            state: TradingState,
            price: Union[float, int, None] = None, 
        ):
            if not price:
                price = self.get_mid_price(product, state)

            self.prices[product] = pd.concat([
                self.prices[product],
                pd.Series({state.timestamp: price})
            ])
        
    def update_past_prices(self,state:TradingState) -> None :
        for product in PRODUCTS:
            mid_price = self.get_mid_price(product,state)
            if mid_price is not None :
                self.past_prices[product].append(mid_price)

    def update_trend(self,state:TradingState,short,long) -> None :
        prices = self.past_prices[PRODUCT2]
        if (len(prices) < long):
            return
        self.trend = 1 if np.mean(prices[-short:]) > np.mean(prices[-long:]) else 0
        

    def save_prices_diving_gear(self, state: TradingState):
        price_diving_gear = self.get_mid_price(DIVING_GEAR, state)
        self.prices[DIVING_GEAR] = pd.concat([
            self.prices[DIVING_GEAR],
            pd.Series({state.timestamp: price_diving_gear})
        ])

    # def get_dolphins_observations(self, state: TradingState):
    #     return state.observations[DOLPHIN_SIGHTINGS]

    # Algorithm logic
    def get_ma_on_product(self,state,product,window):
        prices = self.past_prices[product]
        if (len(prices) < window) :
            return None
        return prices[-window:]





    def pearls_strategy(self, state : TradingState) -> List[Order]:
        """
        Returns a list of orders with trades of PRODUCT1.
        """

        position_pearls = self.get_position(PRODUCT1, state)

        bid_volume = self.position_limit[PRODUCT1] - position_pearls
        ask_volume = - self.position_limit[PRODUCT1] - position_pearls

        orders = []
        orders.append(Order(PRODUCT1, DEFAULT_PRICES[PRODUCT1] - 1, bid_volume))
        orders.append(Order(PRODUCT1, DEFAULT_PRICES[PRODUCT1] + 1, ask_volume))

        return orders

    # def strategy(self,state):
    #     prices = self.past_prices[PRODUCT2]
    #     orders = []
    #     if (len(prices) < 100) : 
    #         return orders
    #     trend = 1 if np.mean(prices[-100:]) < np.mean(prices[-50:]) else 0

    #     if self.trend
        

    def bananas_strategy(self, state : TradingState):
        """
        Returns a list of orders with trades of PRODUCT2.
        """
        orders = []
        prices = self.past_prices[PRODUCT2]
        if (len(prices) < 100) : 
            return orders
        trend = 1 if np.mean(prices[-100:]) < np.mean(prices[-50:]) else 0


        position_bananas = self.get_position(PRODUCT2, state)

        spread = max(state.order_depths[PRODUCT2].buy_orders.keys()) - min(state.order_depths[PRODUCT2].sell_orders.keys())

        bid_volume = self.position_limit[PRODUCT2] - position_bananas
        ask_volume = - self.position_limit[PRODUCT2] - position_bananas


        if position_bananas == 0 and abs(spread) <= 2:
            # Not long nor short
            orders.append(Order(PRODUCT2, math.floor(self.ema_prices[PRODUCT2] - 1), bid_volume))
            orders.append(Order(PRODUCT2, math.ceil(self.ema_prices[PRODUCT2] + 1), ask_volume))
        
        if position_bananas > 0 :
            # Long position
            orders.append(Order(PRODUCT2, math.floor(self.ema_prices[PRODUCT2] - 1), bid_volume))
            orders.append(Order(PRODUCT2, math.ceil(self.ema_prices[PRODUCT2]), ask_volume))

        if position_bananas < 0:
            # Short position
            orders.append(Order(PRODUCT2, math.floor(self.ema_prices[PRODUCT2]), bid_volume))
            orders.append(Order(PRODUCT2, math.ceil(self.ema_prices[PRODUCT2] + 1), ask_volume))

        return orders
    def kelp_strategy(self, state : TradingState):
        """
        Returns a list of orders with trades of PRODUCT2.
        """
        orders = []


        position_bananas = self.get_position(PRODUCT2, state)

        bid_volume = self.position_limit[PRODUCT2] - position_bananas
        ask_volume = - self.position_limit[PRODUCT2] - position_bananas


        sell_orders = state.order_depths[PRODUCT2].sell_orders
        buy_orders = state.order_depths[PRODUCT2].buy_orders


        bid = max(buy_orders.keys())
        ask = min(sell_orders.keys())

        spread = abs(bid-ask)
        
        if (max(sell_orders.keys()) > bid) and spread <= 3:
            orders.append(Order(PRODUCT2,ask,bid_volume))

        # orders.append(Order(PRODUCT2,bid,bid_volume))
        # orders.append(Order(PRODUCT2,ask,ask_volume))

        return orders
    # def bananas_strategy(self, state : TradingState) -> List[Order]:
    #     return orders
    
    # def coconuts_pina_coladas_strategy(self, state : TradingState) -> List[List[Order]]:
    #     return orders_coconuts, orders_pina_coladas
    
    # def coconut_strategy(self, state: TradingState):
    #     return orders

    # def berries_strategy(self, state: TradingState)-> List[Order]:
    #     return order_berries
    
    # def diving_gear_strategy(self, state: TradingState) -> List[Order]:
    #     return orders_diving_gear
    
    # def picnic_strategy(self, state: TradingState)-> List[Order]:
    #     return orders_baguette, orders_basket, orders_dip, orders_ukulele



    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        self.round += 1
        pnl = self.update_pnl(state)
        self.update_ema_prices(state)
        self.update_past_prices(state)
        long = 30
        short = 50

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
        # try:
        #     result[PRODUCT1] = self.pearls_strategy(state)
        # except Exception as e:
        #     logger.print("Error in PRODUCT1 strategy")
        #     logger.print(e)

        # BANANA STRATEGY
        try:
            # result[PRODUCT2] = self.bananas_strategy(state)
            result[PRODUCT2] = self.kelp_strategy(state)
            '''
            orders = []
            prices = self.past_prices[product]
            position_bananas = self.get_position(PRODUCT2,state)
            bid_volume = self.position_limit[PRODUCT2] - position_bananas
            ask_volume = - self.position_limit[PRODUCT2] - position_bananas
            prev_trend = self.trend
            curr_trend = 1 if np.mean(prices[-short:]) > np.mean(prices[-long:]) else 0
            if curr_trend > prev_trend :
                if position_bananas == 0:
                    orders.append(Order(PRODUCT2, math.floor(self.ema_prices[PRODUCT2] - 1), bid_volume))
                    # orders.append(Order(PRODUCT2, math.ceil(self.ema_prices[PRODUCT2] + 1), ask_volume))
                
                if position_bananas > 0 :
                    # Long position
                    orders.append(Order(PRODUCT2, math.floor(self.ema_prices[PRODUCT2] - 1), bid_volume))
                    # orders.append(Order(PRODUCT2, math.ceil(self.ema_prices[PRODUCT2]), ask_volume))

                if position_bananas < 0:
                    # Short position
                    orders.append(Order(PRODUCT2, math.floor(self.ema_prices[PRODUCT2]), bid_volume))
                    # orders.append(Order(PRODUCT2, math.ceil(self.ema_prices[PRODUCT2] + 1), ask_volume))
            elif curr_trend < prev_trend :
                if position_bananas == 0:
                    # orders.append(Order(PRODUCT2, math.floor(self.ema_prices[PRODUCT2] - 1), bid_volume))
                    orders.append(Order(PRODUCT2, math.ceil(self.ema_prices[PRODUCT2] + 1), ask_volume))
                
                if position_bananas > 0 :
                    # Long position
                    # orders.append(Order(PRODUCT2, math.floor(self.ema_prices[PRODUCT2] - 1), bid_volume))
                    orders.append(Order(PRODUCT2, math.ceil(self.ema_prices[PRODUCT2]), ask_volume))

                if position_bananas < 0:
                    # Short position
                    # orders.append(Order(PRODUCT2, math.floor(self.ema_prices[PRODUCT2]), bid_volume))
                    orders.append(Order(PRODUCT2, math.ceil(self.ema_prices[PRODUCT2] + 1), ask_volume))

            result[PRODUCT2] = orders    
            '''
            



        except Exception as e:
            logger.print("Error in PRODUCT2 strategy")
            logger.print(e)

        self.update_trend(state,short,long)
        logger.print("+---------------------------------+",self.trend)
        logger.flush(state, result, conversions=0, trader_data="SAMPLE")

        return result, 0, "SAMPLE"
        