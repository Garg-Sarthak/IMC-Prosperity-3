from typing import Dict, List
import numpy as np
import json
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
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        

        # Initialize the method output dict as an empty dict
        # print(state.traderData)
        result = {}
        if state.traderData == "" :
            traderObj = {}
            for product in state.listings.keys():
                traderObj[f"{product}_state"] = -1
                traderObj[product] = []
            state.traderData = json.dumps(traderObj)
        
        mkt_trades = state.market_trades
        for product in mkt_trades.keys():
            traderObj = json.loads(state.traderData)
            if (len(mkt_trades[product]) != 0):
                # localObj = [{"p" : i.price,"q" : i.quantity} for i in mkt_trades[product]]
                for trade in mkt_trades[product]:
                    traderObj[product].append(trade.price)
            state.traderData = json.dumps(traderObj)

            
        # Iterate over all the keys (the available products) contained in the order dephts
        for product in state.order_depths.keys():


                # Retrieve the Order Depth containing all the market BUY and SELL orders
                order_depth: OrderDepth = state.order_depths[product]
                orders: list[Order] = []

                traderObj = json.loads(state.traderData)
                productObj = traderObj[product]
                if (len(productObj) < 14) : 
                    continue
                
                currPos = state.position[product] if product in state.position.keys() else 0
                # logger.print("curr",currPos)

                mn50 = np.mean(np.array(productObj[-9:]))
                mn100 = np.mean(np.array(productObj[-14:]))
                # logger.print("mean : ",mn)
                std = np.std(np.array(productObj))
                prevState = traderObj[f"{product}_state"]
                currState = 0 if mn50 < mn100 else 1
                traderObj[f"{product}_state"] = currState

                state.traderData = json.dumps(traderObj)




                # if mid > mn : 
                # if mn50 > mn100:
                if prevState == 0 and currState==1:
                    if len(order_depth.sell_orders) > 0:

                        # Sort all the available sell orders by their price,
                        # and select only the sell order with the lowest price
                        best_ask = min(order_depth.sell_orders.keys())
                        best_ask_volume = order_depth.sell_orders[best_ask]
                        # acceptable_price = mn - 0.1*st

                        if currPos == 50 : 
                            continue

                        possibleQty = max(LIMIT - currPos,0)  # 50 - (30) = 20
                        
                        orders.append(Order(product, best_ask, min(best_ask_volume,possibleQty)))
                        # Check if the lowest ask (sell order) is lower than the above defined fair value
                        # if best_ask !=0 :

                            # In case the lowest ask is lower than our fair value,
                            # This presents an opportunity for us to buy cheaply
                            # The code below therefore sends a BUY order at the price level of the ask,
                            # with the same quantity
                            # We expect this order to trade with the sell order
                            # sarthakPrices.append(best_ask)
                            # print("BUY", str(best_ask_volume) + "x", best_ask)
                            #50 - (-10) = 60



                # elif mn50 < mn100 :
                elif prevState == 1 and currState==0:
                    if len(order_depth.buy_orders) != 0:


                        best_bid = max(order_depth.buy_orders.keys())
                        best_bid_volume = order_depth.buy_orders[best_bid]

                        if currPos == -50 :
                            continue
                        possibleQty = max(LIMIT - currPos,0)

                            # print("SELL", str(best_bid_volume) + "x", best_bid)
                        orders.append(Order(product, best_bid,max(best_bid_volume,-1*possibleQty)))
                        # acceptable_price = mn + 0.4*std
                        # if best_bid != 0:
                            # sarthakPrices.append(best_bid)
                            # possibleQty = LIMIT - currPos
                            # # print("SELL", str(best_bid_volume) + "x", best_bid)
                            # orders.append(Order(product, best_bid, possibleQty))
                # The below code block is similar to the one above,
                # the difference is that it find the highest bid (buy order)
                # If the price of the order is higher than the fair value
                # This is an opportunity to sell at a premium

                # Add all the above the orders to the result dict
                result[product] = orders
                
        traderData = "SAMPLE" # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.
        
        conversions = 1 

                # Return the dict of orders
                # These possibly contain buy or sell orders
                # Depending on the logic above
        logger.flush(state, result, conversions, state.traderData)
        return result, conversions, state.traderData

