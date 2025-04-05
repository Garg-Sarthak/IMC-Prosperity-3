from typing import Dict, List
import numpy as np
import json
from datamodel import OrderDepth, TradingState, Order, Trade
# from myTraderData import _update_State


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
                traderObj[product] = []
            state.traderData = json.dumps(traderObj)
        
        mkt_trades = state.market_trades
        for product in mkt_trades.keys():
            traderObj = json.loads(state.traderData)
            if (len(mkt_trades[product]) != 0):
                # localObj = [{"p" : i.price,"q" : i.quantity} for i in mkt_trades[product]]
                for i in mkt_trades[product]:
                    localObj = {"p" : i.price,"q" : i.quantity}
                    traderObj[product].append(localObj)
            state.traderData = json.dumps(traderObj)

            
        # Iterate over all the keys (the available products) contained in the order dephts
        for product in state.order_depths.keys():


                # Retrieve the Order Depth containing all the market BUY and SELL orders
                order_depth: OrderDepth = state.order_depths[product]
                orders: list[Order] = []

                traderObj = json.loads(state.traderData)
                mn = 0
                st = 0
                if (len(traderObj[product]) > 0) :
                    arr = np.array([x['p']*x['q'] for x in traderObj[product]])
                    mn = np.mean(arr)
                    st = np.std(arr)
                
                

                if len(order_depth.sell_orders) > 0:

                    # Sort all the available sell orders by their price,
                    # and select only the sell order with the lowest price
                    best_ask = min(order_depth.sell_orders.keys())
                    best_ask_volume = order_depth.sell_orders[best_ask]
                    if mn!=0 and st/mn < 0.1 :
                        acceptable_price = mn
                    else : 
                        acceptable_price = mn + 0.4*st

                    # Check if the lowest ask (sell order) is lower than the above defined fair value
                    if best_ask <= acceptable_price:

                        # In case the lowest ask is lower than our fair value,
                        # This presents an opportunity for us to buy cheaply
                        # The code below therefore sends a BUY order at the price level of the ask,
                        # with the same quantity
                        # We expect this order to trade with the sell order
                        # sarthakPrices.append(best_ask)
                        # print("BUY", str(best_ask_volume) + "x", best_ask)
                        orders.append(Order(product, best_ask, best_ask_volume))

                # The below code block is similar to the one above,
                # the difference is that it find the highest bid (buy order)
                # If the price of the order is higher than the fair value
                # This is an opportunity to sell at a premium
                if len(order_depth.buy_orders) != 0:
                    best_bid = max(order_depth.buy_orders.keys())
                    best_bid_volume = order_depth.buy_orders[best_bid]
                    if mn!=0 and st/mn < 0.1 :
                        acceptable_price = mn
                    else : 
                        acceptable_price = mn - 0.4*st
                    if best_bid > acceptable_price:
                        # sarthakPrices.append(best_bid)
                        # print("SELL", str(best_bid_volume) + "x", best_bid)
                        orders.append(Order(product, best_bid, best_bid_volume))

                # Add all the above the orders to the result dict
                result[product] = orders
                
        traderData = "SAMPLE" # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.
        
        conversions = 1 

                # Return the dict of orders
                # These possibly contain buy or sell orders
                # Depending on the logic above
        
        return result, conversions, state.traderData

