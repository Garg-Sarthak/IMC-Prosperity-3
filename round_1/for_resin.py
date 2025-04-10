from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict, Any
import string
import jsonpickle
import numpy as np
import math
import json

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


class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    # KELP = "KELP" # Assuming only RAINFOREST_RESIN for this example


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
}


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
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params
        # Position limits for each product
        self.LIMIT = {Product.RAINFOREST_RESIN: 50}  # Max position allowed

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

    # --- take_best_orders, clear_position_order, take_orders, clear_orders (trading logic unchanged) ---
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
                        logger.print(f"TAKE HIT ASK: {product} Price: {best_ask}, Qty: {quantity}")
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
                        logger.print(f"TAKE HIT BID: {product} Price: {best_bid}, Qty: {-quantity}")
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
                logger.print(f"CLEAR HIT BID: {product} Price >= {clear_ask_price}, Qty: {-sent_quantity}")
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
                logger.print(f"CLEAR HIT ASK: {product} Price <= {clear_bid_price}, Qty: {sent_quantity}")
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
            if constrained_buy_quantity < desired_buy_quantity:
                logger.print(f"MAKE BUY VOL LIMIT: {product} Desired: {desired_buy_quantity}, AskVol: {best_ask_volume}, LimitFactor: {volume_limit_factor}, Limited: {constrained_buy_quantity}")

        final_buy_quantity = constrained_buy_quantity
        if final_buy_quantity > 0:
            logger.print(f"MAKE BUY ORDER: {product} Price: {bid}, Qty: {final_buy_quantity}")
            orders.append(Order(product, bid, final_buy_quantity))

        # --- Sell Order Calculation ---
        desired_sell_quantity = position_limit + (position - sell_order_volume)
        constrained_sell_quantity = desired_sell_quantity
        if volume_limit_factor is not None and len(order_depth.buy_orders) > 0:
            best_bid_volume = order_depth.buy_orders[max(order_depth.buy_orders.keys())]
            volume_limit = math.floor(best_bid_volume * volume_limit_factor)
            constrained_sell_quantity = min(desired_sell_quantity, volume_limit)
            if constrained_sell_quantity < desired_sell_quantity:
                logger.print(f"MAKE SELL VOL LIMIT: {product} Desired: {desired_sell_quantity}, BidVol: {best_bid_volume}, LimitFactor: {volume_limit_factor}, Limited: {constrained_sell_quantity}")

        final_sell_quantity = constrained_sell_quantity
        if final_sell_quantity > 0:
            logger.print(f"MAKE SELL ORDER: {product} Price: {ask}, Qty: {-final_sell_quantity}")
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
        logger.print(f"CLEAR Orders: {product} Buys: {buy_order_volume}, Sells: {sell_order_volume}")
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
                logger.print(f"MAKE Ask Join: {product} Fair: {fair_value}, BestAskOut: {best_ask_outside}, JoinEdge: {join_edge}, Ask: {ask}")
            else:
                ask = best_ask_outside - 1
                logger.print(f"MAKE Ask Penny: {product} Fair: {fair_value}, BestAskOut: {best_ask_outside}, Ask: {ask}")
        else:
            logger.print(f"MAKE Ask Default: {product} Fair: {fair_value}, DefaultEdge: {default_edge}, Ask: {ask}")

        bid = round(fair_value - default_edge)
        if best_bid_outside is not None:
            if best_bid_outside >= fair_value - join_edge:
                bid = best_bid_outside
                logger.print(f"MAKE Bid Join: {product} Fair: {fair_value}, BestBidOut: {best_bid_outside}, JoinEdge: {join_edge}, Bid: {bid}")
            else:
                bid = best_bid_outside + 1
                logger.print(f"MAKE Bid Penny: {product} Fair: {fair_value}, BestBidOut: {best_bid_outside}, Bid: {bid}")
        else:
            logger.print(f"MAKE Bid Default: {product} Fair: {fair_value}, DefaultEdge: {default_edge}, Bid: {bid}")

        if manage_position and soft_position_limit > 0:
            current_position = position + buy_order_volume - sell_order_volume
            if current_position > soft_position_limit:
                ask = min(ask, bid + 1)
                logger.print(f"MAKE Pos Adjust Ask: {product} Pos: {current_position}, SoftLimit: {soft_position_limit}, NewAsk: {ask}")
            elif current_position < -soft_position_limit:
                bid = max(bid, ask - 1)
                logger.print(f"MAKE Pos Adjust Bid: {product} Pos: {current_position}, SoftLimit: {-soft_position_limit}, NewBid: {bid}")

        if bid >= ask:
            logger.print(f"MAKE Bid/Ask Invalid: {product} Bid: {bid}, Ask: {ask}. Widening.")
            return orders, buy_order_volume, sell_order_volume

        _, _ = self.market_make(
            product, orders, bid, ask, position,
            buy_order_volume, sell_order_volume,
            order_depth,
            volume_limit_factor
        )

        return orders, buy_order_volume, sell_order_volume
    
    def get_value_on_product(self, product, state : TradingState):
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
                self.past_prices[product] = self.past_prices[product][-1000:]
        return

    def squid_s(self,state:TradingState, product):
        position = state.position.get(product,0)

        orders = []

        buy_qty = self.position_limit[product] - position
        sell_qty = -self.position_limit[product] - position


        market_bids = state.order_depths[product].buy_orders.keys()
        market_asks = state.order_depths[product].sell_orders.keys()

        mid_price = self.get_mid_price(product,state)
        
        best_bid = min(market_bids)
        best_ask = max(market_asks)

        residuals = [p for p in self.past_prices[product][-50:]]
        # std = np.std(residuals) if len(residuals) >= 2 else 0
        # std = np.std([p for p in self.past_prices[product][-50:]])
        std = np.std(residuals)
        mn = np.mean(residuals)
        logger.print(f"mean : {mn}, std : {std}")

        if mid_price >= mn + std:
            orders.append(Order(product,best_ask-1,sell_qty))
            logger.print(f"sell at {best_bid} : {sell_qty}")
        elif mid_price <= mn - std :
            orders.append(Order(product,best_bid+1,buy_qty))
            logger.print(f"buy at {best_ask} : {buy_qty}")

    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData is not None and state.traderData != "":
            try:
                traderObject = jsonpickle.decode(state.traderData)
            except Exception as e:
                logger.print(f"Error decoding traderData: {e}")
                traderObject = {}

        result = {}

        # --- RAINFOREST_RESIN Logic ---
        product = Product.RAINFOREST_RESIN
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

            
            result[product] = resin_take_orders + resin_clear_orders + resin_make_orders

            try:
                result[SQUID] = self.squid_s(state,SQUID)
            except Exception as e:
                logger.print("Error in KELP strategy")
                logger.print(e)

            logger.print(f"Final Orders ({product}): {len(result[product])}")

        conversions = 0
        traderData = jsonpickle.encode(traderObject)

        logger.flush(state, result, conversions, traderData)

        return result, conversions, traderData
