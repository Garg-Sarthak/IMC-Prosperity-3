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

        self.z_score_scaler_for_qty = 4

        


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
                self.past_prices[product] = self.past_prices[product][-1000:]
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

    def resin_s(self,state:TradingState, product):
        position = state.position.get(product,0)

        orders = []

        def_buy = 9997
        def_sell = 10004

        max_position = 50
        # risk_multiplier = 1 - abs(position)/max_position
        risk_multiplier = 1 
        # risk_multiplier = 1
        buy_qty = int((self.position_limit[product] - position)*risk_multiplier)
        sell_qty = int((-self.position_limit[product] - position)*risk_multiplier)


        # market_bids = state.order_depths[product].buy_orders.keys()
        # market_asks = state.order_depths[product].sell_orders.keys()

        mid_price = self.get_mid_price(product,state)
        
        ob = state.order_depths[product]
        best_bid = max(ob.buy_orders.keys()) if ob.buy_orders else None
        best_ask = min(ob.sell_orders.keys()) if ob.sell_orders else None
        if not best_bid or not best_ask: return orders

        # residuals = [p for p in self.past_prices[product][-50:]]
        # std = np.std(residuals)
        # mn = np.mean(residuals)
        price_window = 100
        # long_window = 100
        spread = best_ask - best_bid
        # price_adjustment = (int)(spread*mid_price)
        price_adjustment = 0

        # orders.append(Order(product,def_buy-price_adjustment,buy_qty))
        # orders.append(Order(product,def_sell+price_adjustment,sell_qty))
        orders.append(Order(product,mid_price-spread/2,buy_qty))
        orders.append(Order(product,mid_price+spread/2,sell_qty))


        logger.print(orders)
        return orders

    def squid_s(self,state:TradingState, product):
        position = state.position.get(product,0)

        orders = []

        max_position = 50
        risk_multiplier = 1 - abs(position)/max_position
        # risk_multiplier = 1
        buy_qty = int((self.position_limit[product] - position)*risk_multiplier)
        sell_qty = int((-self.position_limit[product] - position)*risk_multiplier)


        market_bids = state.order_depths[product].buy_orders.keys()
        market_asks = state.order_depths[product].sell_orders.keys()

        mid_price = self.get_mid_price(product,state)
        
        # best_bid = min(market_bids)
        # best_ask = max(market_asks)
        ob = state.order_depths[product]
        best_bid = min(ob.buy_orders.keys()) if ob.buy_orders else None
        best_ask = max(ob.sell_orders.keys()) if ob.sell_orders else None
        if not best_bid or not best_ask: return orders

        # residuals = [p for p in self.past_prices[product][-50:]]
        # std = np.std(residuals)
        # mn = np.mean(residuals)
        price_window = 31
        long_window = 100

        std = np.std(self.past_prices[product][-price_window:])
        mn = np.mean(self.past_prices[product][-price_window:])
        logger.print(f"mean : {mn}, std : {std}")

        spread = min(ob.sell_orders.keys()) - max(ob.buy_orders.keys())
        price_adjustment = max(1, int(spread * 0.2))  # Dynamic tick adjustment

        long_ma = np.mean(self.past_prices[product][-long_window:]) if len(self.past_prices[product]) >=long_window else mn
    
        # if (mid_price >= mn + std) and mid_price < long_ma:
        if (mid_price >= mn + std) :
            order_price = best_ask - price_adjustment # sell
            # order_price = best_bid + price_adjustment
            orders.append(Order(product, order_price, sell_qty))
            
        # elif (mid_price <= mn - std) and mid_price > long_ma:
        elif (mid_price <= mn - std) :
            order_price = best_bid + price_adjustment # buy
            # order_price = best_ask - price_adjustment
            orders.append(Order(product, order_price, buy_qty))
            
        else:
            # Post passive orders
            orders.append(Order(product, best_bid + price_adjustment , buy_qty))  # sell
            orders.append(Order(product, best_ask - price_adjustment , sell_qty))  # Improve bid


        
        # if st is None  or mid_price is None:
        logger.print(orders)
        return orders
        
    def squid_s_2(self, state: TradingState, product):
        position = state.position.get(product, 0)
        orders = []
        
        # 1. Get order book correctly
        ob = state.order_depths[product]
        best_bid = max(ob.buy_orders.keys()) if ob.buy_orders else None
        best_ask = min(ob.sell_orders.keys()) if ob.sell_orders else None
        if not best_bid or not best_ask: return orders
        
        # 2. Calculate fair value and volatility
        mid_price = (best_bid + best_ask)/2
        price_window = 50
        residuals = [p - np.mean(self.past_prices[product][-price_window:]) 
                    for p in self.past_prices[product][-price_window:]]
        std = np.std(residuals) if len(residuals) >= 2 else 0
        mn = np.mean(self.past_prices[product][-price_window:])
        
        # 3. Dynamic order placement
        spread = best_ask - best_bid
        price_adjustment = max(1, int(spread * 0.2))  # 20% of current spread

        # 4. CORRECT ORDER PRICING LOGIC
        if mid_price >= mn + std:
            # SELL: Place below best ask to get priority
            sell_price = best_ask - price_adjustment
            sell_qty = min(-1, (-self.position_limit[product] - position))
            orders.append(Order(product, sell_price, sell_qty))
            
        elif mid_price <= mn - std:
            # BUY: Place above best bid to get priority
            buy_price = best_bid + price_adjustment
            buy_qty = max(1, (self.position_limit[product] - position))
            orders.append(Order(product, buy_price, buy_qty))
        
        # 5. Neutral zone market making
        else:
            # Improve bid/ask spread
            orders.append(Order(product, best_bid + 1, buy_qty))  # Better bid
            orders.append(Order(product, best_ask - 1, sell_qty))  # Better ask

        return orders
        
    def squid_strategy(self,state:TradingState,product):
        position = state.position.get(product,0)

        orders = []

        max_position = 50
        # risk_multiplier = 1 - abs(position)/max_position
        risk_multiplier = 1
        buy_qty = int((self.position_limit[product] - position)*risk_multiplier)
        sell_qty = int((-self.position_limit[product] - position)*risk_multiplier)

        mid_price = self.get_mid_price(product,state)
        
        ob = state.order_depths[product]
        best_bid = max(ob.buy_orders.keys()) if ob.buy_orders else None
        best_ask = min(ob.sell_orders.keys()) if ob.sell_orders else None
        if not best_bid or not best_ask: return orders


        price_window = 31
        short_window = 51
        long_window = 200

        residuals = self.past_prices[product][-short_window:]
        residuals_long = self.past_prices[product][-long_window:]

        mn = np.mean(residuals)
        std = np.std(residuals)

        spread = min(ob.sell_orders.keys()) - max(ob.buy_orders.keys())
        z_score = (mid_price - mn)/std if std else mid_price-mn

        signal_strength = abs(z_score)/3
        buy_qty = buy_qty * signal_strength
        sell_qty = sell_qty * signal_strength


        logger.print(f"mean : {mn}, std : {std}, spread : {spread}, z : {z_score}")

        long_ma = np.mean(self.past_prices[product][-long_window:]) if len(self.past_prices[product]) >=long_window else mn
        long_std = np.std(self.past_prices[product][-long_window:]) if len(self.past_prices[product]) >=long_window else std
        long_z_score = abs((mid_price - long_ma)/long_std) if long_std else 0
        # long_z_score = 0
        

        if (mid_price >= mn + std*1.45) and long_z_score <= 3:

            order_price = best_ask  # sell
            orders.append(Order(product, best_ask, sell_qty))
            logger.print(f"away from mean : sell")
            
        elif (mid_price <= mn - std*1.45) and long_z_score <= 3:
            order_price = best_bid # buy
            orders.append(Order(product, best_bid, buy_qty))
            logger.print(f"away from mean : buy")

        elif long_z_score > 3 :
            if position > 0 and mid_price > long_ma: 
                orders.append(Order(product,best_bid,(self.position_limit[product] - position)))
            elif position < 0 and mid_price < long_ma:
                orders.append(Order(product,best_ask,(-self.position_limit[product] - position)))

        
        logger.print(orders)
        return orders

    def squid_strategy_2(self, state: TradingState, product: str):
        # --- 1. Setup, Get Data, Basic Checks ---
        position = state.position.get(product, 0)
        orders = []
        limit = self.position_limit.get(product, 50)

        mid_price = self.get_mid_price(product, state)
        order_depth = state.order_depths.get(product)
        if mid_price is None or not order_depth: return orders # Early exit if market data invalid

        best_bid = max(order_depth.buy_orders.keys(), default=None)
        best_ask = min(order_depth.sell_orders.keys(), default=None)
        if best_bid is None or best_ask is None or best_ask <= best_bid: return orders # Need valid BBO

        # --- 2. Update History ---
        # self.past_prices.setdefault(product, []).append(mid_price)
        # Optional: Trim history if needed

        # --- 3. Calculate Indicators ---
        short_window = 51 # For MR signal
        long_window = 100 # For Trend Filter

        # Ensure enough history

        residuals_short = self.past_prices[product][-short_window:]
        mn_short = np.mean(residuals_short)
        std_short = np.std( residuals_short)

        residuals_long = self.past_prices[product][-long_window:]
        long_ma = np.mean(residuals_long)

        # Avoid division by zero or unstable signals if std dev is tiny
        if std_short < 1e-6:
            # logger.print("Info: Std dev near zero.")
            # Maybe clear positions passively if std dev collapses? Optional.
            # if position != 0:
            #    clear_price = best_bid if position < 0 else best_ask
            #    orders.append(Order(product, clear_price, -position))
            return orders # Otherwise, do nothing

        z_score = (mid_price - mn_short) / std_short
        spread = best_ask - best_bid

        # --- 4. Calculate Scaled Quantity (Requirement 1 & 2) ---
        # Scale desired trade size based on Z-score magnitude
        # Ensures qty is between 0 and 1.0
        signal_strength_qty = min(1.0, max(0.0, abs(z_score) / self.z_score_scaler_for_qty))

        buy_room = limit - position
        sell_room = -limit - position # This is negative

        # Calculate final quantity, cast to int
        # Ensure we only calculate non-zero entry quantity if a signal is active
        # We calculate exit quantity later
        entry_buy_qty = 0
        entry_sell_qty = 0
        if abs(z_score) >= 1.0: # Only scale entry size if outside +/- 1 std (or your chosen threshold)
             entry_buy_qty = int(round(buy_room * signal_strength_qty))
             entry_sell_qty = int(round(sell_room * signal_strength_qty)) # sell_room is negative

             # Ensure quantities are correctly signed and non-zero for logic below
             entry_buy_qty = max(0, entry_buy_qty)
             entry_sell_qty = min(0, entry_sell_qty)


        # --- 5. Calculate Price Adjustment for Aggressive Entries (Requirement 1c) ---
        # Adjustment logic from your last code: abs(z_score) * spread / 8.0
        # Ensure adjustment is reasonable (e.g., at least 0, maybe cap it?)
        price_adjustment = max(0.0, abs(z_score) * spread / 8.0)
        # Convert to nearest tick (integer) - use round, floor, or ceil
        tick_adjustment = math.ceil(price_adjustment) # Example: Ceiling makes it slightly more aggressive

        # --- 6. Define Signal and Filter States ---
        # Use a threshold slightly > 1.0 if scaling starts at 1.0, or use 1.0 if scaling is zero below 1.0
        entry_threshold_std = 1.0
        sell_signal_active = z_score >= entry_threshold_std
        buy_signal_active = z_score <= -entry_threshold_std
        within_bands = not sell_signal_active and not buy_signal_active

        # Trend filter conditions
        trend_filter_allows_sell = mid_price < long_ma
        trend_filter_allows_buy = mid_price > long_ma

        # --- 7. Order Placement Logic ---

        # A) ENTRY LOGIC (Signal Active + Trend Filter Allows)
        if sell_signal_active and trend_filter_allows_sell:
            if entry_sell_qty < 0: # Check if we have room and calculated qty is non-zero
                intended_sell_price = best_ask - tick_adjustment # Aggressive price
                final_sell_price = max(best_bid + 1, int(round(intended_sell_price))) # Safety clamp & integer price
                orders.append(Order(product, final_sell_price, entry_sell_qty))
                # logger.print(f"SELL ENTRY: Px={final_sell_price}, Qty={entry_sell_qty}, Z={z_score:.2f}, Adj={tick_adjustment}")

        elif buy_signal_active and trend_filter_allows_buy:
             if entry_buy_qty > 0: # Check if we have room and calculated qty is non-zero
                intended_buy_price = best_bid + tick_adjustment # Aggressive price
                final_buy_price = min(best_ask - 1, int(round(intended_buy_price))) # Safety clamp & integer price
                orders.append(Order(product, final_buy_price, entry_buy_qty))
                # logger.print(f"BUY ENTRY: Px={final_buy_price}, Qty={entry_buy_qty}, Z={z_score:.2f}, Adj={tick_adjustment}")

        # B) CLEAR POSITION LOGIC (Within Bands - Requirement 2)
        elif within_bands and position != 0: # Only clear if we have a position
            if position < 0: # If short, place passive buy to close full position
                clear_qty = -position # Positive quantity
                # logger.print(f"CLEAR SHORT: Placing BUY at {best_bid} for {clear_qty}")
                orders.append(Order(product, best_bid, clear_qty))
            elif position > 0: # If long, place passive sell to close full position
                clear_qty = -position # Negative quantity
                # logger.print(f"CLEAR LONG: Placing SELL at {best_ask} for {clear_qty}")
                orders.append(Order(product, best_ask, clear_qty))

        # C) REFRAIN LOGIC (Implicit - Requirement 3)
        # No 'else' block needed here. If outside bands but trend filter blocks entry,
        # AND not within bands, no orders are generated for entry or clearing.

        # --- 8. Return Orders ---
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
        #     result[RESIN] = self.resin_s(state,RESIN)
        # except Exception as e:
        #     logger.print("Error in RESIN strategy")
        #     logger.print(e)

        # # KELP STRATEGY
        # try:
        #     result[KELP] = self.squid_s(state,KELP)
        # except Exception as e:
        #     logger.print("Error in KELP strategy")
        #     logger.print(e)

        try:
            result[SQUID] = self.squid_strategy(state,SQUID)
        except Exception as e:
            logger.print("Error in squid strategy")
            logger.print(e)

        logger.print("+---------------------------------+")
        logger.flush(state, result, conversions=0, trader_data="SAMPLE")

        return result, 0, "SAMPLE"