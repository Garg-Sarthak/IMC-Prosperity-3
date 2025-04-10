import pandas as pd
import numpy as np # For NaN handling if needed

# --- 1. Load the Data ---
file_path = '/Users/sarthak/Desktop/imc/round_2/round-2-island-data-bottle_2/prices_round_2_day_-1.csv' # <<< CHANGE THIS
print(f"Loading data from: {file_path}")
try:
    day_1 = pd.read_csv(file_path, delimiter=';')
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    exit() # Or handle appropriately
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit()

print("Data loaded.")

# --- 2. Keep Only Necessary Columns ---
# We only need timestamp, product, and the price we want to use (mid_price)
prices_df = day_1[['timestamp', 'product', 'mid_price']].copy()

# --- 3. Ensure 'mid_price' is Numeric ---
# Sometimes prices might be read as strings; convert them to numbers.
# 'coerce' turns any values that can't be converted into NaN (Not a Number)
prices_df['mid_price'] = pd.to_numeric(prices_df['mid_price'], errors='coerce')

# --- 4. Pivot the Table ---
# This rearranges the table so:
# - Index is the 'timestamp'
# - Columns are the unique 'product' names
# - Values are the 'mid_price' for that product at that timestamp
print("Pivoting table to align prices by timestamp...")
try:
    # This assumes one price per product per timestamp. If duplicates exist:
    # Option 1: Keep the last price: prices_df = prices_df.drop_duplicates(subset=['timestamp', 'product'], keep='last')
    # Option 2: Average prices: prices_df = prices_df.groupby(['timestamp', 'product']).mid_price.mean().reset_index()
    # Let's assume unique for now, or use drop_duplicates if needed:
    prices_df = prices_df.drop_duplicates(subset=['timestamp', 'product'], keep='last')
    aligned_prices = prices_df.pivot(index='timestamp', columns='product', values='mid_price')
except Exception as e:
    print(f"Error pivoting data: {e}")
    print("This might happen if there are duplicate (timestamp, product) pairs.")
    print("Try uncommenting the 'drop_duplicates' line above the pivot call.")
    exit()


# --- 5. Fill Missing Prices ---
# When one product updates but others don't, the pivot table has gaps (NaN).
# 'ffill' (forward fill) fills NaN with the previous known value for that product.
# 'bfill' (backward fill) fills NaNs at the start.
print("Filling gaps in price data (forward fill)...")
aligned_prices = aligned_prices.bfill().ffill() # Use both for robustness

# --- 6. Define Basket Compositions ---
BASKET1_COMPONENTS = {"CROISSANT": 6, "JAM": 3, "DJEMBE": 1}
BASKET2_COMPONENTS = {"CROISSANT": 4, "JAM": 2}
BASKET1_NAME = "PICNIC_BASKET1"
BASKET2_NAME = "PICNIC_BASKET2"

# --- 7. Calculate Synthetic Values and Spreads ---
# Now we can directly use the columns in the 'aligned_prices' DataFrame

print("Calculating synthetic values and spreads...")

# Basket 1
required_cols_b1 = list(BASKET1_COMPONENTS.keys()) + [BASKET1_NAME]
if all(col in aligned_prices.columns for col in required_cols_b1):
    print("Processing Basket 1...")
    aligned_prices['Synth_PB1'] = (
        aligned_prices['CROISSANT'] * BASKET1_COMPONENTS['CROISSANT'] +
        aligned_prices['JAM'] * BASKET1_COMPONENTS['JAM'] +
        aligned_prices['DJEMBE'] * BASKET1_COMPONENTS['DJEMBE']
    )
    aligned_prices['Spread_PB1'] = aligned_prices[BASKET1_NAME] - aligned_prices['Synth_PB1']
else:
    print(f"Warning: Missing one or more columns needed for Basket 1 calculation: {required_cols_b1}")
    aligned_prices['Synth_PB1'] = np.nan
    aligned_prices['Spread_PB1'] = np.nan

# Basket 2
required_cols_b2 = list(BASKET2_COMPONENTS.keys()) + [BASKET2_NAME]
if all(col in aligned_prices.columns for col in required_cols_b2):
    print("Processing Basket 2...")
    aligned_prices['Synth_PB2'] = (
        aligned_prices['CROISSANT'] * BASKET2_COMPONENTS['CROISSANT'] +
        aligned_prices['JAM'] * BASKET2_COMPONENTS['JAM']
    )
    aligned_prices['Spread_PB2'] = aligned_prices[BASKET2_NAME] - aligned_prices['Synth_PB2']
else:
     print(f"Warning: Missing one or more columns needed for Basket 2 calculation: {required_cols_b2}")
     aligned_prices['Synth_PB2'] = np.nan
     aligned_prices['Spread_PB2'] = np.nan


# --- 8. Show Results ---
print("\n--- Calculation Results (first 5 rows) ---")
# Display relevant columns
display_cols = [BASKET1_NAME, 'Synth_PB1', 'Spread_PB1', BASKET2_NAME, 'Synth_PB2', 'Spread_PB2']
# Filter display_cols to only those that actually exist in the dataframe
display_cols = [col for col in display_cols if col in aligned_prices.columns]
print(aligned_prices[display_cols].head())

print("\n--- Spread Statistics ---")
if 'Spread_PB1' in aligned_prices.columns:
    print("\nSpread_PB1 Stats:")
    print(aligned_prices['Spread_PB1'].describe())
if 'Spread_PB2' in aligned_prices.columns:
    print("\nSpread_PB2 Stats:")
    print(aligned_prices['Spread_PB2'].describe())

# Now you have 'aligned_prices' DataFrame containing the spreads
# You can use aligned_prices['Spread_PB1'] and aligned_prices['Spread_PB2']
# for your mean reversion analysis and strategy development.