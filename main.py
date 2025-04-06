import csv
from datamodel import TradingState, Listing, Observation, OrderDepth
from MA_Crossover import Trader

def load_csv_data(csv_file_path):
    data = []
    with open(csv_file_path, mode='r') as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            # print(row)
            data.append(row)
    return data


# # # Load data from the CSV file
csv_file = "prices_round_1_day_0.csv"
csv_data = load_csv_data(csv_file)
# print(csv_data)

# # # Process CSV rows into order depths.
order_depths = {}
for row in csv_data:
    product = row["product"]

    # If product is not yet in order_depths, initialize it
    if product not in order_depths:
        order_depths[product] = OrderDepth()
        order_depths[product].buy_orders = {}
        order_depths[product].sell_orders = {}

    # Extract bid and ask price/volume levels from the row
    for i in range(1, 4):  # There are 3 bid/ask levels in the CSV
        bid_price_key = f"bid_price_{i}"
        bid_volume_key = f"bid_volume_{i}"
        ask_price_key = f"ask_price_{i}"
        ask_volume_key = f"ask_volume_{i}"

        # Parse and add bid orders (buy orders)
        if row[bid_price_key] and row[bid_volume_key]:  # Check if they exist
            bid_price = int(row[bid_price_key])
            bid_volume = int(row[bid_volume_key])
            order_depths[product].buy_orders[bid_price] = bid_volume

        # Parse and add ask orders (sell orders)
        if row[ask_price_key] and row[ask_volume_key]:  # Check if they exist
            ask_price = int(row[ask_price_key])
            ask_volume = int(row[ask_volume_key])
            order_depths[product].sell_orders[ask_price] = ask_volume

# Create listings for each product found in the CSV
    listings = {product: Listing(product, product, "USD") for product in order_depths.keys()}

    # Create a TradingState using the order depths and listings derived from the CSV.
    state = TradingState(
        traderData="",
        timestamp=int(csv_data[0]["timestamp"]),  # Use first row timestamp
        listings=listings,
        order_depths=order_depths,
        own_trades={},
        market_trades={},
        position={},
        observations=Observation({}, {})
    )

    # # # Initialize the trader and run it with the created state
    trader = Trader()
    result, conversions, trader_data = trader.run(state)

    print("Orders:", result)
    print("Conversions:", conversions)
    print("Trader data:", trader_data)
    # break

# import csv
# import sys # For potential exit on error
# from typing import Dict, List, Tuple # Added for type hinting clarity

# # Assuming these are defined in your 'datamodel.py'
# # Make sure datamodel.py is in the same directory or accessible in PYTHONPATH
# try:
#     from datamodel import TradingState, Listing, OrderDepth, Observation, Trade, Symbol, Product, Position
#     # Assuming Order class is needed for the Trader's return value
#     # If Order is not in datamodel, define it or import it from where it exists
#     class Order:
#         def __init__(self, symbol: Symbol, price: int, quantity: int):
#             self.symbol = symbol
#             self.price = price
#             self.quantity = quantity # Positive for buy, negative for sell
#         def __repr__(self):
#             # Provide a useful string representation for printing
#             return f"Order({self.symbol}, {self.price}, {self.quantity})"

# except ImportError:
#     print("Error: Could not import classes from 'datamodel'. Make sure 'datamodel.py' is accessible.")
#     # Define dummy classes to allow script structure analysis, but it won't fully run
#     Symbol = str
#     Product = str
#     Position = int
#     class Listing: pass
#     class OrderDepth:
#         def __init__(self): self.buy_orders = {}; self.sell_orders = {}
#     class Observation:
#          def __init__(self, plainValueObservations, conversionObservations): pass
#     class Trade: pass
#     class TradingState: pass
#     class Order:
#         def __init__(self, symbol, price, quantity): self.symbol, self.price, self.quantity = symbol, price, quantity
#         def __repr__(self): return f"Order({self.symbol}, {self.price}, {self.quantity})"
#     print("Warning: Using dummy datamodel classes.")


# # Assuming your Trader class is defined in 'trade.py'
# # Make sure trade.py is in the same directory or accessible in PYTHONPATH
# try:
#     from trade import Trader
# except ImportError:
#     print("Error: Could not import 'Trader' from 'trade'. Make sure 'trade.py' exists.")
#     # Define a dummy Trader for testing the script structure
#     class Trader:
#         def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
#             print(f"Dummy Trader received state for timestamp: {getattr(state, 'timestamp', 'N/A')}")
#             # Default behavior: do nothing
#             return {}, 0, "" # orders, conversions, traderData
#     print("Warning: Using dummy Trader class.")


# def load_csv_data(csv_file_path: str) -> List[Dict[str, str]]:
#     """Loads data from a semicolon-delimited CSV file."""
#     data = []
#     try:
#         with open(csv_file_path, mode='r', encoding='utf-8') as f:
#             reader = csv.DictReader(f, delimiter=";")
#             if not reader.fieldnames:
#                  print(f"Error: CSV file '{csv_file_path}' appears to be empty or header is missing.")
#                  return []
#             # Basic header check - adjust if your headers differ
#             expected_headers = ["timestamp", "product", "bid_price_1", "ask_price_1"]
#             if not all(h in reader.fieldnames for h in expected_headers):
#                  print(f"Warning: CSV headers might be missing expected columns. Found: {reader.fieldnames}")
#             for row in reader:
#                 data.append(row)
#     except FileNotFoundError:
#         print(f"Error: CSV file not found at '{csv_file_path}'")
#         return []
#     except Exception as e:
#         print(f"Error reading CSV file '{csv_file_path}': {e}")
#         return []
#     return data

# # --- Main Script Logic ---

# csv_file = "prices_round_1_day_0.csv"  # Make sure this path is correct
# all_csv_data = load_csv_data(csv_file)

# if not all_csv_data:
#     print("Failed to load or process CSV data. Exiting.")
#     sys.exit(1) # Exit if we can't load data

# # --- Find the latest timestamp in the data ---
# latest_timestamp = -1
# try:
#     # Ensure timestamps are treated as integers for comparison
#     timestamps = [int(row['timestamp']) for row in all_csv_data if row.get('timestamp', '').isdigit()]
#     if timestamps:
#         latest_timestamp = max(timestamps)
#     else:
#         print("Error: No valid numeric timestamps found in the CSV data.")
#         sys.exit(1)
# except KeyError:
#     print("Error: 'timestamp' column not found in CSV.")
#     sys.exit(1)
# except ValueError:
#     print("Error: Could not convert timestamp values to integers.")
#     sys.exit(1)

# print(f"Found latest timestamp in CSV: {latest_timestamp}")

# # --- Filter data for ONLY the latest timestamp ---
# data_for_latest_timestamp = [row for row in all_csv_data if row.get('timestamp') == str(latest_timestamp)] # Compare as string if loaded as string initially

# if not data_for_latest_timestamp:
#     # This case should ideally not happen if latest_timestamp was derived correctly
#     print(f"Error: No data rows found for the latest timestamp {latest_timestamp}. Check CSV integrity.")
#     sys.exit(1)

# # --- Process the filtered data (only rows for the latest timestamp) ---
# order_depths: Dict[Symbol, OrderDepth] = {}
# listings: Dict[Symbol, Listing] = {}
# processed_products = set()

# print(f"Processing {len(data_for_latest_timestamp)} rows for timestamp {latest_timestamp}...")

# for row in data_for_latest_timestamp:
#     product = row.get("product")
#     if not product:
#         print(f"Warning: Skipping row with missing product at timestamp {latest_timestamp}: {row}")
#         continue

#     # Track products processed at this timestamp
#     processed_products.add(product)

#     # Create Listing for the product if not already done
#     if product not in listings:
#         # Assuming denomination is Seashells based on context, adjust if needed
#         listings[product] = Listing(symbol=product, product=product, denomination="SEASHELLS")

#     # Create OrderDepth for the product if not already done
#     if product not in order_depths:
#         order_depths[product] = OrderDepth()
#         # Explicitly initialize the dictionaries (good practice)
#         order_depths[product].buy_orders = {}
#         order_depths[product].sell_orders = {}

#     # Extract bid and ask price/volume levels from this product's row
#     # Important: Reset orders for *this product* before populating from its row
#     current_buy_orders = {}
#     current_sell_orders = {}

#     for i in range(1, 4):  # Assuming max 3 levels (bid_price_1, bid_price_2, etc.)
#         bid_price_key = f"bid_price_{i}"
#         bid_volume_key = f"bid_volume_{i}"
#         ask_price_key = f"ask_price_{i}"
#         ask_volume_key = f"ask_volume_{i}"

#         # Parse and add bid orders (buy orders for the bots -> people can sell to them)
#         if row.get(bid_price_key) and row.get(bid_volume_key):
#             try:
#                 # Use float conversion first to handle potential decimals before int
#                 bid_price = int(float(row[bid_price_key]))
#                 bid_volume = int(float(row[bid_volume_key]))
#                 if bid_volume > 0:  # Only add orders with positive volume
#                     current_buy_orders[bid_price] = bid_volume
#             except (ValueError, TypeError):
#                  # print(f"Warning: Invalid bid data for {product} level {i} at ts {latest_timestamp}. Skipping.")
#                  pass # Optionally print warning

#         # Parse and add ask orders (sell orders for the bots -> people can buy from them)
#         if row.get(ask_price_key) and row.get(ask_volume_key):
#             try:
#                 ask_price = int(float(row[ask_price_key]))
#                 ask_volume = int(float(row[ask_volume_key]))
#                 if ask_volume > 0:  # Only add orders with positive volume
#                     # Store asks with POSITIVE volume in OrderDepth, as per documentation
#                     current_sell_orders[ask_price] = ask_volume
#             except (ValueError, TypeError):
#                 # print(f"Warning: Invalid ask data for {product} level {i} at ts {latest_timestamp}. Skipping.")
#                 pass # Optionally print warning

#     # Assign the extracted orders to the OrderDepth object for this product
#     order_depths[product].buy_orders = current_buy_orders
#     order_depths[product].sell_orders = current_sell_orders


# # --- Create the TradingState ---
# # Initialize other state components as empty for this single snapshot test
# initial_positions: Dict[Product, Position] = {prod: 0 for prod in processed_products}
# initial_own_trades: Dict[Symbol, List[Trade]] = {prod: [] for prod in processed_products}
# initial_market_trades: Dict[Symbol, List[Trade]] = {prod: [] for prod in processed_products}

# # Assuming a basic Observation structure for now
# initial_observations = Observation({}, {}) # Empty plainValue and conversionObservations


# # Check if all necessary classes were imported/defined before creating TradingState
# if 'TradingState' in locals() and callable(TradingState):
#     state = TradingState(
#         traderData="",  # No prior state data for the first run
#         timestamp=latest_timestamp,
#         listings=listings,
#         order_depths=order_depths,
#         own_trades=initial_own_trades,
#         market_trades=initial_market_trades,
#         position=initial_positions, # Start with zero positions
#         observations=initial_observations
#     )

#     # --- Initialize the trader and run it ---
#     trader = Trader() # Use the imported or dummy Trader

#     print("\n--- Calling Trader.run ---")
#     # Use try-except block to catch errors during trader execution
#     try:
#         result_orders, result_conversions, result_trader_data = trader.run(state)
#         print("--- Trader.run Finished Successfully ---")

#         # --- Print the results ---
#         print("\n--- Results from Trader ---")
#         print("Orders Submitted by Trader:")
#         if result_orders:
#             for product, orders_list in result_orders.items():
#                 # Use repr for potentially better formatting of Order objects
#                 print(f"  {product}: {[repr(o) for o in orders_list]}")
#         else:
#             print("  (No orders submitted)")

#         print(f"Conversions Requested: {result_conversions}")
#         print(f"Returned Trader Data: '{result_trader_data}'")

#         # Optional: Print generated state details for verification
#         # print("\n--- Generated TradingState Details ---")
#         # print(f"Timestamp: {state.timestamp}")
#         # print("Listings:", state.listings.keys())
#         # print("Initial Positions:", state.position)
#         # print("Order Depths:")
#         # for product, depth in state.order_depths.items():
#         #     sorted_buy = dict(sorted(depth.buy_orders.items(), reverse=True))
#         #     sorted_sell = dict(sorted(depth.sell_orders.items()))
#         #     print(f"  {product}: Buy={sorted_buy}, Sell={sorted_sell}")

#     except Exception as e:
#         print(f"\n--- Error during Trader.run execution ---")
#         print(f"Error: {e}")
#         import traceback
#         traceback.print_exc() # Print detailed traceback

# else:
#     print("Error: TradingState class not defined. Cannot create state.")
#     sys.exit(1)

