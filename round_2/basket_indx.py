from typing import Dict, List
import numpy as np
import json
import math
from datamodel import OrderDepth, TradingState, Order, Trade
# from scipy.stats import linregress
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

BASKETS = [
    BASKET1,
    BASKET2
]

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


class Trader:

    def __init__(self) -> None:
        
        logger.print("Initializing Trader...")

        self.position_limit = {
            RESIN : 50,
            KELP : 50,
            SQUID : 50,
            CROISSANTS: 250,
            JAMS: 350,
            DJEMBES: 60,
            BASKET1 : 60,
            BASKET2 : 90
        }


        # self.past_prices keeps the list of all past prices
        self.past_prices = dict()
        for product in PRODUCTS:
            self.past_prices[product] = []

        # self.ema_prices keeps an exponential moving average of prices
        self.ema_prices = dict()
        for product in PRODUCTS:
            self.ema_prices[product] = None

        self.past_sperad = dict()
        for basket in BASKETS:
            self.past_sperad[basket] = []

  


    def get_position(self, product, state : TradingState):
        return state.position.get(product, 0)    

    def get_mid_price(self, product, state : TradingState):

        market_bids = state.order_depths[product].buy_orders
        market_asks = state.order_depths[product].sell_orders
        
        best_bid = max(market_bids)
        best_ask = min(market_asks)
        return (best_bid + best_ask)/2

    def get_swmid(self,product,state):
        market_bids = state.order_depths[product].buy_orders
        market_asks = state.order_depths[product].sell_orders

        best_bid = max(market_bids)
        best_ask = min(market_asks)
        bid_vol = market_bids[best_bid]
        ask_vol = market_asks[best_ask]

        return (best_bid*bid_vol + best_ask*ask_vol)/(bid_vol+ask_vol)


    def update_past_prices(self,state:TradingState) -> None :
        for product in PRODUCTS:
            mid_price = self.get_swmid(product,state)
            if mid_price is not None :
                self.past_prices[product].append(mid_price)
        return
    
    def calculate_sma(self, prices: List[float], period: int):

        if period <= 0:
             logger.print(f"Error: SMA period must be positive (got {period}).")
             return None
        if len(prices) < period:
            # Not enough historical data to calculate the SMA for the given period
            return None
        else:
            # Select the most recent 'period' prices and calculate their mean
            sma_value = np.mean(prices[-period:])
            return sma_value

    
    def get_trend_slope(prices, window=10):
            if len(prices) < window:
                return 0
            x = list(range(window))
            y = prices[-window:]
            slope, _, r_value, _, _ = linregress(x, y)
            return slope * r_value  # slope * correlation gives strength + direction
    
    def strat2(self, state: TradingState, product):
        position = state.position.get(product, 0)
        orders = []

        buy_qty = self.position_limit[product] - position
        sell_qty = -self.position_limit[product] - position

        market_bids = state.order_depths[product].buy_orders.keys()
        market_asks = state.order_depths[product].sell_orders.keys()

        mid_price = self.get_mid_price(product, state)

        best_bid = max(market_bids)
        best_ask = min(market_asks)

        residuals = [p for p in self.past_prices[product][-50:]]
        recent_prices = self.past_prices[product][-10:]

        std = np.std(residuals)
        mn = np.mean(residuals)
        slope = recent_prices[-1] - recent_prices[0]    # delta of prices
        threshold = std * 3.2   # random value - to tune

        logger.print(f"mean : {mn}, std : {std}, slope: {slope}, threshold : {threshold}")

        # === Trendy Phase ===

        #going up
        if slope > threshold:
            if position >= 0:
                # Exit longs if you're already holding
                orders.append(Order(product, best_ask, -position))
                logger.print(f"Market trending UP - Sell holdings at {best_ask}")
            else:
                # Mini momentum scalp - trend up
                scalp_qty = min(20,50 - position )
                if scalp_qty > 0:
                    orders.append(Order(product, best_bid + 1, scalp_qty))
                    logger.print(f"Scalp BUY {scalp_qty} at {best_bid + 1} on UP momentum")

        elif slope < -threshold:
            if position <= 0:
                # Exit shorts
                orders.append(Order(product, best_bid,  position))
                logger.print(f"Market trending DOWN - Buy back at {best_bid}")
            else:
                # Mini momentum scalp - trend down
                scalp_qty = min(-50-position,-20)
                if scalp_qty > 0:
                    orders.append(Order(product, best_ask - 1, scalp_qty))
                    logger.print(f"Scalp SELL {scalp_qty} at {best_ask - 1} on DOWN momentum")

        # === Ranging Market ===
        else:
            if mid_price >= mn + std*1.45 :
                orders.append(Order(product, best_ask, sell_qty))
                logger.print(f"sell at {best_bid} : {sell_qty}")
            elif mid_price <= mn - std*1.45:
                orders.append(Order(product, best_bid, buy_qty))
                logger.print(f"buy at {best_ask} : {buy_qty}")

        return orders

    def update_past_spread(self,state:TradingState):
        for basket in BASKETS:
            synthetic_price = 0
            for pair in BASKET1_WEIGHTS.keys():
                synthetic_price += BASKET1_WEIGHTS[pair] * self.get_swmid(pair,state)
            self.past_spreads[basket].append(synthetic_price)

    def get_synthetic_basket_order_depth(self,state:TradingState,basket):
        synthetic_order_price = OrderDepth()
        params = BASKET_PARAMS[basket]

        implied_bid = 0
        implied_ask = 0

        best_asks = {}
        best_bids = {}

        for component in params["components"]:
            best_asks[component] = min(state.order_depths[component].sell_orders.keys())
            best_bids[component] = max(state.order_depths[component].buy_orders.keys())

        for product in params["components"]:
            implied_bid += params["weights"][product] * best_bids[product]
            implied_ask += params["weights"][product] * best_asks[product]
        
        implied_bid_volume = float("inf")
        if implied_bid > 0:
            for component in params["components"]:
                implied_bid_volume = min(implied_bid_volume, state.order_depths[component].buy_orders[best_bids[component]]//params["weights"][component])
            synthetic_order_price.buy_orders[implied_bid] = implied_bid_volume

        implied_ask_volume = float("inf")
        if implied_ask < float("inf"):
            for component in params["components"]:
                implied_ask_volume = min(implied_ask_volume, -state.order_depths[component].sell_orders[best_asks[component]]//params["weights"][component])
            synthetic_order_price.sell_orders[implied_ask] = -implied_ask_volume
        
        return synthetic_order_price

    # Add this function inside your Trader class in basket_indx.py

    def convert_synthetic_orders_to_component_orders(
        self,
        basket_name: str, # ADDED: To know which basket's params to use
        synthetic_orders: List[Order],
        state: TradingState # CHANGED: Pass state to access all order depths
    ) -> Dict[str, List[Order]]:
        """
        Converts hypothetical orders for a synthetic basket into actual,
        executable orders for its individual components based on BBO prices.
        """
        # Get parameters for the specified basket
        if basket_name not in BASKET_PARAMS:
             logger.print(f"Error: Basket parameters not found for {basket_name}")
             return {}
        params = BASKET_PARAMS[basket_name]
        components = params["components"]
        weights = params["weights"]

        # Initialize the dictionary to store component orders
        component_orders = {comp: [] for comp in components} # Initialize for expected components

        # Get the implied best bid and ask for the synthetic basket
        # Ensure your get_synthetic_basket_order_depth function is available in 'self'
        # and works correctly using the basket_name
        synthetic_basket_order_depth = self.get_synthetic_basket_order_depth(state, basket_name)

        # Handle case where synthetic depth couldn't be calculated
        if not synthetic_basket_order_depth:
             logger.print(f"Cannot convert orders for {basket_name}: Synthetic depth unavailable.")
             return {}

        synth_best_bid = max(synthetic_basket_order_depth.buy_orders.keys()) if synthetic_basket_order_depth.buy_orders else 0
        synth_best_ask = min(synthetic_basket_order_depth.sell_orders.keys()) if synthetic_basket_order_depth.sell_orders else float('inf')

        # Iterate through each synthetic basket order provided
        for order in synthetic_orders:
            # Extract the price and quantity from the synthetic basket order
            synth_price = order.price # Price the synthetic order was notionally placed at
            synth_quantity = order.quantity # Quantity of synthetic baskets

            component_exec_prices = {}
            can_execute = True

            # Determine component execution prices based on synthetic order direction
            if synth_quantity > 0: # Buying synthetic -> Buy components at their ASKS
                if synth_price < synth_best_ask: # Check if aggressive enough
                      logger.print(f"Warning: Synthetic BUY order price {synth_price} below synthetic ask {synth_best_ask} for {basket_name}. Skipping.")
                      continue # Skip if not aggressive enough relative to implied BBO

                for component in components:
                    comp_od = state.order_depths.get(component)
                    if not comp_od or not comp_od.sell_orders:
                        logger.print(f"Error converting for {basket_name}: Component {component} has no sell orders.")
                        can_execute = False
                        break
                    component_exec_prices[component] = min(comp_od.sell_orders.keys()) # Component best ASK

            elif synth_quantity < 0: # Selling synthetic -> Sell components at their BIDS
                if synth_price > synth_best_bid: # Check if aggressive enough
                     logger.print(f"Warning: Synthetic SELL order price {synth_price} above synthetic bid {synth_best_bid} for {basket_name}. Skipping.")
                     continue # Skip if not aggressive enough relative to implied BBO

                for component in components:
                    comp_od = state.order_depths.get(component)
                    if not comp_od or not comp_od.buy_orders:
                        logger.print(f"Error converting for {basket_name}: Component {component} has no buy orders.")
                        can_execute = False
                        break
                    component_exec_prices[component] = max(comp_od.buy_orders.keys()) # Component best BID
            else: # quantity is zero
                continue # Skip zero quantity orders

            if not can_execute:
                continue # Skip this synthetic order if any component price was missing

            # Create orders for each component
            for component in components:
                 if component not in component_exec_prices: continue # Should not happen if can_execute is True

                 comp_price = component_exec_prices[component]
                 comp_weight = weights.get(component, 0) # Get weight for this component
                 if comp_weight == 0: continue # Skip if weight is 0

                 # Calculate component quantity, maintaining sign from synthetic order
                 comp_quantity = synth_quantity * comp_weight

                 # Create the component order object
                 component_order = Order(component, comp_price, comp_quantity)
                 component_orders[component].append(component_order)

        return component_orders





    def basket_1(self,state:TradingState,product,basket):
        orders = []

        mean_spread = BASKET_PARAMS[product]["mean_spread"]
        std_dev_window = BASKET_PARAMS[product]["std_dev_window"]

        params = BASKET_PARAMS[product]

        std_dev_spread = np.std(self.past_spreads[product][-std_dev_window:])

        curr_pos = state.position.get(product,0)

        curr_sythetic_price = 0
        for pair in BASKET1_WEIGHTS.keys():
            curr_sythetic_price += BASKET1_WEIGHTS[pair] * self.get_swmid(pair,state)

        z_score_spread = (curr_sythetic_price - mean_spread) / std_dev_spread

        target_pos = 0

        basket_price = self.get_swmid(product,state)

        if z_score_spread > params["z_entry_threshold"]:
            target_pos = -params["target_position"]
        elif z_score_spread < -params["z_entry_threshold"]:
            target_pos = params["target_position"]

        self.get_synthetic_basket_order_depth

   
    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        # self.round += 1


        self.update_past_prices(state)
        self.update_past_spread(state)
        # Initialize the method output dict as an empty dict
        result = {}

        # # KELP STRATEGY
        # try:
        #     result[KELP] = self.squid_s(state,KELP)
        # except Exception as e:
        #     logger.print("Error in KELP strategy")
        #     logger.print(e)

        try:
            result[SQUID] = self.strat2(state,SQUID)
        except Exception as e:
            logger.print("Error in squid strategy")
            logger.print(e)

        logger.print("+---------------------------------+")
        logger.flush(state, result, conversions=0, trader_data="SAMPLE")

        return result, 0, "SAMPLE"