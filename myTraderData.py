from datamodel import OrderDepth, TradingState, Order, Trade
import json

def getTrades(self, state:TradingState,product) -> any:
    market_trade : list[Trade] = state.market_trades[product]

    trades : list[Trade] = [{trade.timestamp : [trade.price,trade.quantity]} for trade in market_trade]
    # print (trades)
    # print(trades)
    return trades


def _update_State(self,state:TradingState,product):
    
    traderString : str = state.traderData

    # market_trade : list[Trade] = state.market_trades
    # print("tji",market_trade)
    # trades = []
    # for trade in market_trade : 
    #     currD = {}
    #     currD[trade.timestamp] = {
    #         "price" : trade.price,
    #         "quantity" : trade.quantity
    #     }
    #     if (len(trades) < 2):
    #         print (trades)
    #     trades.append(trade)

    # # trades : list[Trade] = [{trade.timestamp : [trade.price,trade.quantity]} for trade in market_trade]

    # traderObj : dict =  json.loads(traderString)
    # newTradeData = json.dumps(traderObj)
    # print("nts",newTradeData)
    # return newTradeData
