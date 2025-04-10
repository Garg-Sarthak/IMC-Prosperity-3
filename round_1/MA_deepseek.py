from typing import Dict, List
import numpy as np
import json
import math
from datamodel import OrderDepth, TradingState, Order, Trade

from typing import Any
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

# from myTraderData import _update_State

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


LIMIT = 50

class Trader:
    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        result = {}
        trader_data = json.loads(state.traderData) if state.traderData else {"prices": {}, "state": {}}

        # Update price history for each product
        # for product in state.market_trades:
        #     if product not in trader_data["prices"]:
        #         trader_data["prices"][product] = []
        #     # Append new trade prices (max 20 to avoid memory bloat)
        #     new_prices = [trade.price for trade in state.market_trades[product]]
        #     trader_data["prices"][product].extend(new_prices)
        #     trader_data["prices"][product] = trader_data["prices"][product][-20:]



        for product in state.order_depths:
            if product not in trader_data["prices"]:
                trader_data["prices"][product] = []
            sell_orders = state.order_depths[product].sell_orders  # {price: -qty}
            buy_orders = state.order_depths[product].buy_orders    # {price: +qty}

            # Use absolute values for sell quantities
            sellSum = sum(price for price, qty in sell_orders.items())
            sellQty = sum(abs(qty) for qty in sell_orders.values())
            buySum = sum(price for price, qty in buy_orders.items())
            buyQty = sum(buy_orders.values())

            md = (sellSum + buySum) / (len(sell_orders.items())+len(buy_orders.items()))

            # Avoid division by zero
            # if (sellQty + buyQty) == 0:
            #     md = 0  # Or fallback to last traded price
            # else:
            # print ("md",md)
            trader_data["prices"][product].append(md)
            trader_data["prices"][product] = trader_data["prices"][product][-200:]  # Keep last 200 data points



            order_depth = state.order_depths[product]
            orders = []
            curr_pos = state.position.get(product, 0)
            
            # Get price history and ensure enough data
            price_history = trader_data["prices"].get(product, [])
            if len(price_history) < 200:
                continue  # Wait for sufficient data
            
            # Dynamic MA periods based on recent volatility
            rolling_std = np.std(price_history) 
            rolling_mean = np.mean(price_history)
            period = 200
            # rolling_std = (np.mean(price_history[-period:])**2) + (np.mean([i**2 for i in price_history[-period:]])) # 10-period volatility
            # rolling_std = math.sqrt(rolling_std)
            # rolling_std = (md-np.mean(price_history[-period:]))/(rolling_std)
            rolling_std /= rolling_mean
            # rolling_std = 0
            logger.print(rolling_std,"rolling")
            ma_short = np.mean(price_history[-51:]) 
            ma_long = np.mean(price_history[-200:])
            
            prev_state = trader_data["state"].get(product, -1)
            curr_state = 1 if ma_short > ma_long else 0
            
            # Generate signals
            if prev_state == 0 and curr_state == 1:  # Buy signal
                if order_depth.sell_orders:
                    best_ask = min(order_depth.sell_orders.keys())
                    best_bid = max(order_depth.buy_orders.keys())
                    best_ask_vol = abs(order_depth.sell_orders[best_ask])  # Fix: Use absolute value
                    max_possible_buy = LIMIT - curr_pos
                    buy_vol = min(best_ask_vol, max_possible_buy)
                    if buy_vol > 0:     
                        orders.append(Order(product, best_ask, buy_vol))  # Positive quantity
            
            elif prev_state == 1 and curr_state == 0:  # Sell signal
                if order_depth.buy_orders:
                    best_bid = max(order_depth.buy_orders.keys())
                    best_ask = min(order_depth.sell_orders.keys())
                    best_bid_vol = order_depth.buy_orders[best_bid]  # Already positive
                    max_possible_sell = LIMIT + curr_pos  # Corrected formula
                    sell_vol = -min(best_bid_vol, max_possible_sell)  # Negative quantity
                    if sell_vol < 0:    
                        orders.append(Order(product, best_bid, sell_vol))  # Negative quantity
            
            
            trader_data["state"][product] = curr_state
            result[product] = orders
        
        trader_data_str = json.dumps(trader_data)
        logger.flush(state, result, conversions=0, trader_data=trader_data_str)
        return result, 0, trader_data_str