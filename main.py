import csv
from datamodel import TradingState, Listing, Observation, OrderDepth
from trade import Trader

def load_csv_data(csv_file_path):
    data = []
    with open(csv_file_path, mode='r') as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            data.append(row)
    return data

# Load data from the CSV file
csv_file = "prices_round_1_day_0.csv"
csv_data = load_csv_data(csv_file)

# Process CSV rows into order depths.
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

# Initialize the trader and run it with the created state
trader = Trader()
result, conversions, trader_data = trader.run(state)

print("Orders:", result)
print("Conversions:", conversions)
print("Trader data:", trader_data)
