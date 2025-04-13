
from typing import Dict, List
import numpy as np
import json
import jsonpickle
import math
from typing import Any, Tuple
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
BASKET1 = "PICNIC_BASKET1"
BASKET2 = "PICNIC_BASKET2"
CROISSANTS = "CROISSANTS"
DJEMBES = "DJEMBES"
JAMS = "JAMS"

PRODUCTS = [
    KELP,
    RESIN,
    SQUID,
    BASKET1,
    BASKET2,
    CROISSANTS,
    DJEMBES,
    JAMS
]

DEFAULT_PRICES = {
    RESIN : 10000,
    KELP : 2016,
    SQUID : 2040,
    BASKET1 : 2040,
    BASKET2 : 2040,
    CROISSANTS : 0,
    DJEMBES : 0,
    JAMS : 0
}

BASKET1_WEIGHTS = {
    CROISSANTS : 6,
    JAMS : 3,
    DJEMBES : 1
}

BASKET2_WEIGHTS = {
    CROISSANTS : 4,
    JAMS : 2
}

BASKET_PARAMS = {
    BASKET1: {
        "mean_spread": 48.0,         # Your observed mean
        "std_dev_window": 30,        # Short window (e.g., 20-50)
        "z_entry_threshold": 1.5,    # Z-score to trigger entry
        "z_exit_threshold": 0.5,     # Z-score threshold to trigger exit towards 0
        "target_position": 60,       # Target absolute position (slightly below limit 60)
        "components": [CROISSANTS, JAMS, DJEMBES],
        "weights": BASKET1_WEIGHTS,
        "limit": 60, # Store limit here for easy access
        "hist_key": "spread_hist_pb1" # Key for traderData
    },
    BASKET2: {
        "mean_spread": 30.0,         # Your observed mean
        "std_dev_window": 30,        # Short window (e.g., 20-50)
        "z_entry_threshold": 1.5,    # Z-score to trigger entry
        "z_exit_threshold": 0.5,     # Z-score threshold to trigger exit towards 0
        "target_position": 90,       # Target absolute position (slightly below limit 100)
        "components": [CROISSANTS, JAMS],
        "weights": BASKET2_WEIGHTS,
        "limit": 100, # Store limit here
        "hist_key": "spread_hist_pb2" # Key for traderData
    },
}

# Component limits (needed for execution checks)
COMPONENT_LIMITS = {
    CROISSANTS: 250,
    JAMS: 350,
    DJEMBES: 60,
}



class Trader:

    def __init__(self) -> None:
        
        logger.print("Initializing Trader...")

        self.ema_prices = {product: None for product in PRODUCTS}
        self.ema_param = 0.33

        # Position Limits (consolidated)
        self.position_limit = {
            RESIN : 50, # Example, confirm R1/R2 limits
            KELP : 50, # Example, confirm R1/R2 limits
            SQUID : 50, # Example, confirm R1/R2 limits
            **COMPONENT_LIMITS, # Add component limits
            BASKET1 : BASKET_PARAMS[BASKET1]["limit"],
            BASKET2 : BASKET_PARAMS[BASKET2]["limit"]
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
        # self.ema_prices = dict()
        # for product in PRODUCTS:
        #     self.ema_prices[product] = None

        # self.ema_param = 0.33

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


        
        # if st is None  or mid_price is None:
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
        
    def basket_strategy(self, basket_name: str, state: TradingState, trader_data_dict: Dict) -> Dict[str, List[Order]]:
        """Generic function to run the basket strategy."""
        params = BASKET_PARAMS[basket_name]
        orders_to_place = {}

        # 1. Check data availability
        basket_od = state.order_depths.get(basket_name)
        if not basket_od or not basket_od.buy_orders or not basket_od.sell_orders:
            logger.print(f"No valid order depth for {basket_name}")
            return orders_to_place

        component_mid_prices = {}
        for comp in params["components"]:
            comp_mid = self.get_mid_price(comp, state)
            if comp_mid is None:
                logger.print(f"Missing mid price for component {comp} in {basket_name}")
                return orders_to_place # Cannot calculate synthetic without all components
            component_mid_prices[comp] = comp_mid

        # 2. Calculate Prices and Spread
        basket_mid_price = (max(basket_od.buy_orders.keys()) + min(basket_od.sell_orders.keys())) / 2
        synthetic_value = sum(component_mid_prices[comp] * params["weights"][comp] for comp in params["components"])
        current_spread = basket_mid_price - synthetic_value

        # 3. Update History & Calculate Rolling Stats
        spread_history = trader_data_dict.get(params["hist_key"], [])
        spread_history.append(current_spread)
        # Limit history length (use window size + buffer)
        max_hist_len = params["std_dev_window"] * 2
        if len(spread_history) > max_hist_len:
             spread_history = spread_history[-max_hist_len:] # Keep recent history
        trader_data_dict[params["hist_key"]] = spread_history # Update trader_data

        if len(spread_history) < params["std_dev_window"]:
            logger.print(f"Not enough spread history for {basket_name} ({len(spread_history)}/{params['std_dev_window']})")
            return orders_to_place # Not enough data

        rolling_std_dev = np.std(spread_history[-params["std_dev_window"]:])
        if rolling_std_dev < 1e-5: # Avoid division by zero/tiny std dev
            logger.print(f"Spread std dev too low for {basket_name}")
            return orders_to_place

        # 4. Calculate Z-Score
        z_score = (current_spread - params["mean_spread"]) / rolling_std_dev
        logger.print(f"{basket_name}: Spread={current_spread:.2f}, Mean={params['mean_spread']:.2f}, Std={rolling_std_dev:.2f}, Z={z_score:.2f}")

        # 5. Determine Target Position based on Z-score and current position
        current_pos = self.get_position(basket_name, state)
        target_pos = 0 # Default target is flat

        if z_score > params["z_entry_threshold"]:
            target_pos = -params["target_position"] # Target short position
        elif z_score < -params["z_entry_threshold"]:
            target_pos = params["target_position"]  # Target long position
        elif current_pos > 0 and z_score > -params["z_exit_threshold"]: # Exit long
             target_pos = 0
        elif current_pos < 0 and z_score < params["z_exit_threshold"]: # Exit short
             target_pos = 0
        else:
            target_pos = current_pos # Maintain current position if within exit bands or no entry signal


        # 6. Calculate Trade if Target Position Changed
        if target_pos != current_pos:
            trade_quantity = target_pos - current_pos
            logger.print(f"{basket_name}: Current Pos={current_pos}, Target Pos={target_pos}, Trade Qty={trade_quantity}")

            if trade_quantity == 0: return orders_to_place # Should not happen, but safety check

            # Determine execution prices (Aggressive: cross the spread)
            basket_exec_price = min(basket_od.sell_orders.keys()) if trade_quantity > 0 else max(basket_od.buy_orders.keys())
            component_exec_prices = {}
            component_ods = {}
            for comp in params["components"]:
                comp_od = state.order_depths.get(comp)
                if not comp_od or not comp_od.buy_orders or not comp_od.sell_orders:
                    logger.print(f"Cannot execute {basket_name}: Missing BBO for component {comp}")
                    return orders_to_place # Cannot execute without BBO
                component_ods[comp] = comp_od
                # Sell component if buying basket, Buy component if selling basket
                component_exec_prices[comp] = max(comp_od.buy_orders.keys()) if trade_quantity > 0 else min(comp_od.sell_orders.keys())

            # 7. Calculate Max Volume & Check Limits
            trade_abs_qty = abs(trade_quantity)
            basket_avail_vol = abs(basket_od.sell_orders[basket_exec_price]) if trade_quantity > 0 else abs(basket_od.buy_orders[basket_exec_price])
            component_avail_vols = []
            component_limit_max_size = []

            current_comp_positions = {comp: self.get_position(comp, state) for comp in params["components"]}

            for comp in params["components"]:
                comp_weight = params["weights"][comp]
                comp_od = component_ods[comp]
                comp_exec_price = component_exec_prices[comp]
                comp_trade_sign = -1 if trade_quantity > 0 else 1 # Sign of component trade
                comp_pos_change = comp_trade_sign * trade_abs_qty * comp_weight

                # Check BBO volume
                if comp_trade_sign == -1: # Selling component
                    avail_vol = abs(comp_od.buy_orders.get(comp_exec_price, 0))
                else: # Buying component
                     avail_vol = abs(comp_od.sell_orders.get(comp_exec_price, 0))

                if avail_vol < trade_abs_qty * comp_weight:
                    component_avail_vols.append(math.floor(avail_vol / comp_weight)) # Max baskets possible based on this component's volume
                else:
                    component_avail_vols.append(float('inf')) # Not limited by volume

                # Check position limits
                new_comp_pos = current_comp_positions[comp] + comp_pos_change
                comp_limit = self.position_limit.get(comp, 0) # Use .get for safety
                if abs(new_comp_pos) > comp_limit:
                     # Calculate max size allowed by this component's limit
                    if comp_pos_change > 0: # Position increasing
                        allowed_change = comp_limit - current_comp_positions[comp]
                    else: # Position decreasing (more negative)
                         allowed_change = -comp_limit - current_comp_positions[comp] # e.g. -250 - (-200) = -50 allowed

                    max_size = math.floor(abs(allowed_change) / comp_weight) if comp_weight > 0 else float('inf')
                    component_limit_max_size.append(max_size)
                else:
                     component_limit_max_size.append(float('inf')) # Not limited by position

            # Check basket position limit
            basket_limit = params["limit"]
            new_basket_pos = current_pos + trade_quantity
            basket_limit_max_size = float('inf')
            if abs(new_basket_pos) > basket_limit:
                 if trade_quantity > 0:
                      allowed_change = basket_limit - current_pos
                 else:
                      allowed_change = -basket_limit - current_pos
                 basket_limit_max_size = abs(allowed_change)


            # Determine actual trade size
            max_volume_allowed = min(basket_avail_vol, *component_avail_vols)
            max_limit_allowed = min(basket_limit_max_size, *component_limit_max_size)

            actual_trade_size = math.floor(min(trade_abs_qty, max_volume_allowed, max_limit_allowed))

            if actual_trade_size <= 0:
                 logger.print(f"{basket_name}: Trade blocked. Required={trade_abs_qty}, VolLimit={max_volume_allowed:.0f}, PosLimit={max_limit_allowed:.0f}")
                 return orders_to_place

            final_trade_quantity = actual_trade_size if trade_quantity > 0 else -actual_trade_size
            logger.print(f"{basket_name}: EXECUTING TRADE -> Qty: {final_trade_quantity}, BasketPx: {basket_exec_price}")


            # 8. Generate Orders
            orders_to_place[basket_name] = [Order(basket_name, basket_exec_price, final_trade_quantity)]
            for comp in params["components"]:
                comp_weight = params["weights"][comp]
                comp_exec_price = component_exec_prices[comp]
                comp_trade_quantity = (-1 if trade_quantity > 0 else 1) * actual_trade_size * comp_weight # Opposite sign to basket trade
                if comp not in orders_to_place: orders_to_place[comp] = []
                orders_to_place[comp].append(Order(comp, comp_exec_price, comp_trade_quantity))
                logger.print(f"  Component {comp}: Qty: {comp_trade_quantity}, Px: {comp_exec_price}")


        return orders_to_place

    def basket_1_strategy(self, state: TradingState, trader_data_dict: Dict) -> Dict[str, List[Order]]:
        return self.basket_strategy(BASKET1, state, trader_data_dict)

    def basket_2_strategy(self, state: TradingState, trader_data_dict: Dict) -> Dict[str, List[Order]]:
        return self.basket_strategy(BASKET2, state, trader_data_dict)


    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]: # Added Tuple typing
        self.round += 1
        logger.print(f"--- Round {self.round}, Timestamp {state.timestamp} ---")

        # Load traderData
        trader_data_dict = {}
        if state.traderData:
            try:
                trader_data_dict = jsonpickle.decode(state.traderData)
                # Ensure lists exist for history tracking
                if not isinstance(trader_data_dict, dict): trader_data_dict = {} # Reset if invalid format
                if BASKET_PARAMS[BASKET1]["hist_key"] not in trader_data_dict or not isinstance(trader_data_dict[BASKET_PARAMS[BASKET1]["hist_key"]], list):
                     trader_data_dict[BASKET_PARAMS[BASKET1]["hist_key"]] = []
                if BASKET_PARAMS[BASKET2]["hist_key"] not in trader_data_dict or not isinstance(trader_data_dict[BASKET_PARAMS[BASKET2]["hist_key"]], list):
                     trader_data_dict[BASKET_PARAMS[BASKET2]["hist_key"]] = []

            except Exception as e:
                logger.print(f"Error decoding traderData: {e}. Resetting.")
                trader_data_dict = { # Initialize fresh
                    BASKET_PARAMS[BASKET1]["hist_key"]: [],
                    BASKET_PARAMS[BASKET2]["hist_key"]: []
                 }
        else: # Initialize if first run or empty
             trader_data_dict = {
                 BASKET_PARAMS[BASKET1]["hist_key"]: [],
                 BASKET_PARAMS[BASKET2]["hist_key"]: []
             }


        # Update EMA prices (useful for defaults)
        self.update_ema_prices(state)

        # Initialize final result dictionary
        result = {}
        conversions = 0 # Not used in this strategy

        # --- Execute Basket Strategies ---
        try:
            pb1_orders = self.basket_1_strategy(state, trader_data_dict)
            # Merge orders carefully, handling potential key overlaps if components are traded elsewhere
            for product, orders in pb1_orders.items():
                if product not in result: result[product] = []
                result[product].extend(orders)
        except Exception as e:
            logger.print(f"ERROR in Basket 1 Strategy: {e}")

        try:
            pb2_orders = self.basket_2_strategy(state, trader_data_dict)
            # Merge orders
            for product, orders in pb2_orders.items():
                if product not in result: result[product] = []
                result[product].extend(orders)
        except Exception as e:
            logger.print(f"ERROR in Basket 2 Strategy: {e}")


        # --- Execute Other Strategies (Placeholders) ---
        # try:
        #     resin_orders = self.resin_s(state)
        #     if RESIN not in result: result[RESIN] = []
        #     result[RESIN].extend(resin_orders)
        # except Exception as e:
        #     logger.print(f"Error in RESIN strategy: {e}")
        #
        # try:
        #     kelp_orders = self.kelp_s(state) # Assuming you have a kelp strategy
        #     if KELP not in result: result[KELP] = []
        #     result[KELP].extend(kelp_orders)
        # except Exception as e:
        #     logger.print(f"Error in KELP strategy: {e}")
        #
        # try:
        #     squid_orders = self.squid_s(state)
        #     if SQUID not in result: result[SQUID] = []
        #     result[SQUID].extend(squid_orders)
        # except Exception as e:
        #     logger.print(f"Error in SQUID strategy: {e}")


        # Encode traderData for next round
        trader_data_encoded = jsonpickle.encode(trader_data_dict)

        # logger.print("Final Orders:", self.compress_orders(result)) # Log compressed final orders
        logger.flush(state, result, conversions, trader_data_encoded)

        return result, conversions, trader_data_encoded

    # def run(self, state: TradingState) -> Dict[str, List[Order]]:
    #     """
    #     Only method required. It takes all buy and sell orders for all symbols as an input,
    #     and outputs a list of orders to be sent
    #     """
    #     self.round += 1
    #     pnl = self.update_pnl(state)
    #     self.update_ema_prices(state)
    #     self.update_past_prices(state)

    #     logger.print(f"Log round {self.round}")

    #     logger.print("TRADES:")
    #     for product in state.own_trades:
    #         for trade in state.own_trades[product]:
    #             if trade.timestamp == state.timestamp - 100:
    #                 logger.print(trade)

    #     logger.print(f"\tCash {self.cash}")
    #     for product in PRODUCTS:
    #         logger.print(f"\tProduct {product}, Position {self.get_position(product, state)}, Midprice {self.get_mid_price(product, state)}, Value {self.get_value_on_product(product, state)}, EMA {self.ema_prices[product]}")
    #     logger.print(f"\tPnL {pnl}")
        

    #     # Initialize the method output dict as an empty dict
    #     result = {}

    #     #  RESIN STRATEGY
    #     # try:
    #     #     result[RESIN] = self.resin_s(state,RESIN)
    #     # except Exception as e:
    #     #     logger.print("Error in RESIN strategy")
    #     #     logger.print(e)

    #     # # KELP STRATEGY
    #     # try:
    #     #     result[KELP] = self.squid_s(state,KELP)
    #     # except Exception as e:
    #     #     logger.print("Error in KELP strategy")
    #     #     logger.print(e)

    #     # try:
    #     #     result[SQUID] = self.squid_s(state,SQUID)
    #     # except Exception as e:
    #     #     logger.print("Error in squid strategy")
    #     #     logger.print(e)

    #     logger.print("+---------------------------------+")
    #     logger.flush(state, result, conversions=0, trader_data="SAMPLE")

    #     return result, 0, "SAMPLE"

##