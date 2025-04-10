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

        self.ema_param = 0.33

        self.all_positions = set()


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

    def update_past_prices(self,state:TradingState) -> None :
        for product in PRODUCTS:
            mid_price = self.get_mid_price(product,state)
            if mid_price is not None :
                self.past_prices[product].append(mid_price)
                self.past_prices[product] = self.past_prices[product][-600:]
        return

    def spread_filling(self, state: TradingState, product):
        orders = []
        position_product = self.get_position(product, state)
        order_depth = state.order_depths[product]

        

        # Extract best bid and best ask from the market
        if order_depth.buy_orders:
            best_bid = min(order_depth.buy_orders.keys())
        else:
            best_bid = None

        if order_depth.sell_orders:
            best_ask = max(order_depth.sell_orders.keys())
        else:
            best_ask = None

        # Use a simple fair price: midpoint of best bid and ask

        # Calculate order volumes based on position
        bid_volume = self.position_limit[product] - position_product
        ask_volume = -self.position_limit[product] - position_product

        # Spread capture logic
        if best_bid is not None and bid_volume > 0:
            buy_price = best_bid + 0
            orders.append(Order(product, buy_price, ask_volume))

        if best_ask is not None and ask_volume < 0:
            sell_price = best_ask - 0
            orders.append(Order(product, sell_price, bid_volume))

        return orders

    def squid_s(self,state:TradingState, product):
        position = state.position.get(product,0)

        orders = []

        max_position = 50
        risk_multiplier = 1 - abs(position)/max_position
        buy_qty = int((self.position_limit[product] - position)*risk_multiplier)
        sell_qty = int((-self.position_limit[product] - position)*risk_multiplier)


        market_bids = state.order_depths[product].buy_orders.keys()
        market_asks = state.order_depths[product].sell_orders.keys()

        mid_price = self.get_mid_price(product,state)
        
        # best_bid = min(market_bids)
        # best_ask = max(market_asks)
        ob = state.order_depths[product]
        best_bid = max(ob.buy_orders.keys()) if ob.buy_orders else None
        best_ask = min(ob.sell_orders.keys()) if ob.sell_orders else None
        if not best_bid or not best_ask: return orders

        # residuals = [p for p in self.past_prices[product][-50:]]
        # std = np.std(residuals)
        # mn = np.mean(residuals)
        price_window = 21
        long_window = 300

        residuals = [p - np.mean(self.past_prices[product][-price_window:]) 
                 for p in self.past_prices[product][-price_window:]]
        # std = np.std(residuals) if len(residuals) >= 2 else 0
        std = np.std(self.past_prices[product][-price_window:])
        mn = np.mean(self.past_prices[product][-price_window:])
        # long_ma =np.mean(self.past_prices[product][-200:])
        logger.print(f"mean : {mn}, std : {std}")

        spread = min(ob.sell_orders.keys()) - max(ob.buy_orders.keys())
        price_adjustment = max(1, int(spread * 0.4))  # Dynamic tick adjustment
        # price_adjustment = max(1, int(spread * 0.4))  # Dynamic tick adjustment

        # if mid_price >= mn + std:
        #     orders.append(Order(product,best_ask,sell_qty))
        #     logger.print(f"sell at {best_bid} : {sell_qty}")
        # elif mid_price <= mn - std :
        #     orders.append(Order(product,best_bid,buy_qty))
            # logger.print(f"buy at {best_ask} : {buy_qty}")

        long_ma = np.mean(self.past_prices[product][-long_window:]) if len(self.past_prices[product]) >=long_window else mn
    
    # 6. Trading Logic with Improvements
        # if (mid_price >= mn + std) :
        if (mid_price >= mn + std) and long_ma < mid_price:
            order_price = best_ask  # sell
            orders.append(Order(product, best_bid+price_adjustment, sell_qty))
            
        elif (mid_price <= mn - std) and long_ma > mid_price:
            order_price = best_bid
            orders.append(Order(product, best_ask-price_adjustment, buy_qty))
            
        else:
            # Post passive orders
            orders.append(Order(product, best_ask + price_adjustment , sell_qty))  # Improve ask
            orders.append(Order(product, best_bid - price_adjustment , buy_qty))  # Improve bid


        
        # if st is None  or mid_price is None:
        logger.print(orders)
        return orders
        

        




    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        self.round += 1
        pnl = self.update_pnl(state)
        self.update_ema_prices(state)
        self.update_past_prices(state)

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

        #  RESIN STRATEGY
        # try:
        #     result[RESIN] = self.resin_strategy(state)
        # except Exception as e:
        #     logger.print("Error in RESIN strategy")
        #     logger.print(e)

        # # KELP STRATEGY
        # try:
        #     result[KELP] = self.kelp_strategy(state)
        # except Exception as e:
        #     logger.print("Error in KELP strategy")
        #     logger.print(e)

        try:
            result[SQUID] = self.squid_s(state,SQUID)
        except Exception as e:
            logger.print("Error in squid strategy")
            logger.print(e)

        logger.print("+---------------------------------+")
        logger.flush(state, result, conversions=0, trader_data="SAMPLE")

        return result, 0, "SAMPLE"