from datamodel import OrderDepth, UserId, TradingState, Order, ConversionObservation
from typing import List, Dict, Any
import numpy as np
import math
from math import log, sqrt, exp
from statistics import NormalDist
import jsonpickle

class Product:
    VOLCANIC_ROCK = "VOLCANIC_ROCK"
    VOLCANIC_ROCK_VOUCHER_9500 = "VOLCANIC_ROCK_VOUCHER_9500"
    VOLCANIC_ROCK_VOUCHER_9750 = "VOLCANIC_ROCK_VOUCHER_9750"
    VOLCANIC_ROCK_VOUCHER_10000 = "VOLCANIC_ROCK_VOUCHER_10000"
    VOLCANIC_ROCK_VOUCHER_10250 = "VOLCANIC_ROCK_VOUCHER_10250"
    VOLCANIC_ROCK_VOUCHER_10500 = "VOLCANIC_ROCK_VOUCHER_10500"

PARAMS = {
    Product.VOLCANIC_ROCK_VOUCHER_9500: {
        "mean_volatility": 0.18,
        "threshold": 0.00163,
        "strike": 9500,
        "std_window": 10,
        "zscore_threshold": 2.0
    },
    Product.VOLCANIC_ROCK_VOUCHER_9750: {
        "mean_volatility": 0.18,
        "threshold": 0.00163,
        "strike": 9750,
        "std_window": 10,
        "zscore_threshold": 2.0
    },
    Product.VOLCANIC_ROCK_VOUCHER_10000: {
        "mean_volatility": 0.18,
        "threshold": 0.00163,
        "strike": 10000,
        "std_window": 10,
        "zscore_threshold": 2.0
    },
    Product.VOLCANIC_ROCK_VOUCHER_10250: {
        "mean_volatility": 0.18,
        "threshold": 0.00163,
        "strike": 10250,
        "std_window": 10,
        "zscore_threshold": 2.0
    },
    Product.VOLCANIC_ROCK_VOUCHER_10500: {
        "mean_volatility": 0.18,
        "threshold": 0.00163,
        "strike": 10500,
        "std_window": 10,
        "zscore_threshold": 2.0
    }
}

class BlackScholes:
    @staticmethod
    def black_scholes_call(spot, strike, time_to_expiry, volatility):
        d1 = (log(spot/strike) + (0.5 * volatility * volatility) * time_to_expiry) / (volatility * sqrt(time_to_expiry))
        d2 = d1 - volatility * sqrt(time_to_expiry)
        call_price = spot * NormalDist().cdf(d1) - strike * NormalDist().cdf(d2)
        return call_price

    @staticmethod
    def delta(spot, strike, time_to_expiry, volatility):
        d1 = (log(spot/strike) + (0.5 * volatility * volatility) * time_to_expiry) / (volatility * sqrt(time_to_expiry))
        return NormalDist().cdf(d1)

    @staticmethod
    def implied_volatility(call_price, spot, strike, time_to_expiry, max_iterations=200, tolerance=1e-10):
        low_vol = 0.01
        high_vol = 1.0
        volatility = (low_vol + high_vol) / 2.0  # Initial guess
        for _ in range(max_iterations):
            estimated_price = BlackScholes.black_scholes_call(spot, strike, time_to_expiry, volatility)
            diff = estimated_price - call_price
            if abs(diff) < tolerance:
                break
            elif diff > 0:
                high_vol = volatility
            else:
                low_vol = volatility
            volatility = (low_vol + high_vol) / 2.0
        return volatility

class Trader:
    def __init__(self):
        self.LIMIT = {
            Product.VOLCANIC_ROCK: 400,
            Product.VOLCANIC_ROCK_VOUCHER_9500: 200,
            Product.VOLCANIC_ROCK_VOUCHER_9750: 200,
            Product.VOLCANIC_ROCK_VOUCHER_10000: 200,
            Product.VOLCANIC_ROCK_VOUCHER_10250: 200,
            Product.VOLCANIC_ROCK_VOUCHER_10500: 200,
        }
        self.params = PARAMS

    def get_mid_price(self, product: str, state: TradingState) -> float:
        order_depth = state.order_depths.get(product, OrderDepth())
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        return (max(order_depth.buy_orders.keys()) + min(order_depth.sell_orders.keys())) / 2

    def voucher_orders(self, voucher_product: str, order_depth: OrderDepth, position: int, trader_data: Dict[str, Any], volatility: float) -> List[Order]:
        if f"past_vol_{voucher_product}" not in trader_data:
            trader_data[f"past_vol_{voucher_product}"] = []
        
        vol_history = trader_data[f"past_vol_{voucher_product}"]
        vol_history.append(volatility)
        
        if len(vol_history) > self.params[voucher_product]["std_window"]:
            vol_history.pop(0)
        
        if len(vol_history) < self.params[voucher_product]["std_window"]:
            return []
        
        vol_zscore = (volatility - self.params[voucher_product]["mean_volatility"]) / np.std(vol_history)
        threshold = self.params[voucher_product]["zscore_threshold"]
        
        orders = []
        if vol_zscore > threshold and position > -self.LIMIT[voucher_product]:
            # Volatility is high, sell vouchers
            if order_depth.buy_orders:
                best_bid = max(order_depth.buy_orders.keys())
                qty = min(self.LIMIT[voucher_product] + position, order_depth.buy_orders[best_bid])
                if qty > 0:
                    orders.append(Order(voucher_product, best_bid, -qty))
        
        elif vol_zscore < -threshold and position < self.LIMIT[voucher_product]:
            # Volatility is low, buy vouchers
            if order_depth.sell_orders:
                best_ask = min(order_depth.sell_orders.keys())
                qty = min(self.LIMIT[voucher_product] - position, -order_depth.sell_orders[best_ask])
                if qty > 0:
                    orders.append(Order(voucher_product, best_ask, qty))
        
        return orders

    def hedge_delta(self, rock_order_depth: OrderDepth, rock_position: int, total_delta: float) -> List[Order]:
        target_rock_position = -int(total_delta)
        hedge_qty = target_rock_position - rock_position
        
        orders = []
        if hedge_qty > 0:
            # Buy VOLCANIC_ROCK
            if rock_order_depth.sell_orders:
                best_ask = min(rock_order_depth.sell_orders.keys())
                qty = min(hedge_qty, -rock_order_depth.sell_orders[best_ask], self.LIMIT[Product.VOLCANIC_ROCK] - rock_position)
                if qty > 0:
                    orders.append(Order(Product.VOLCANIC_ROCK, best_ask, qty))
        
        elif hedge_qty < 0:
            # Sell VOLCANIC_ROCK
            if rock_order_depth.buy_orders:
                best_bid = max(rock_order_depth.buy_orders.keys())
                qty = min(-hedge_qty, rock_order_depth.buy_orders[best_bid], self.LIMIT[Product.VOLCANIC_ROCK] + rock_position)
                if qty > 0:
                    orders.append(Order(Product.VOLCANIC_ROCK, best_bid, -qty))
        
        return orders

    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        # Initialize or decode trader data
        trader_data = {}
        if state.traderData:
            trader_data = jsonpickle.decode(state.traderData)
        
        result = {}
        conversions = 0
        
        # Get volcanic rock price
        rock_price = self.get_mid_price(Product.VOLCANIC_ROCK, state)
        if not rock_price:
            return {}, 0, jsonpickle.encode(trader_data)
        
        # Calculate days to expiry (rounds 1-5 represent days, 7 days to expiry at the start)
        days_to_expiry = max(7 - state.timestamp//1000000, 1)/365  # Convert to years
        
        # Calculate total delta across all voucher positions
        total_delta = 0
        rock_position = state.position.get(Product.VOLCANIC_ROCK, 0)
        
        # Process each voucher
        for voucher_product in self.params:
            # Skip if not in order depths
            if voucher_product not in state.order_depths:
                continue
                
            order_depth = state.order_depths[voucher_product]
            position = state.position.get(voucher_product, 0)
            
            # Get voucher mid price
            voucher_price = self.get_mid_price(voucher_product, state)
            if not voucher_price:
                continue
            
            # Calculate implied volatility
            strike = self.params[voucher_product]["strike"]
            volatility = BlackScholes.implied_volatility(voucher_price, rock_price, strike, days_to_expiry)
            
            # Generate orders based on volatility z-score
            orders = self.voucher_orders(voucher_product, order_depth, position, trader_data, volatility)
            if orders:
                result[voucher_product] = orders
            
            # Calculate and accumulate delta
            delta = BlackScholes.delta(rock_price, strike, days_to_expiry, volatility)
            total_delta += position * delta
        
        # Delta hedge with volcanic rock
        if abs(total_delta) > 0.1:  # Only hedge if delta is significant
            rock_orders = self.hedge_delta(state.order_depths[Product.VOLCANIC_ROCK], rock_position, total_delta)
            if rock_orders:
                result[Product.VOLCANIC_ROCK] = rock_orders
        
        return result, conversions, jsonpickle.encode(trader_data)
