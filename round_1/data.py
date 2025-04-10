import random
import pandas as pd

# Define latitude/longitude boundaries for your region
LAT_MIN, LAT_MAX = 12.90, 12.99
LON_MIN, LON_MAX = 77.55, 77.65

# Occupancy levels
occupancy_levels = ['low', 'medium', 'high']

def generate_data(n=1000):
    data = {
        'day_of_year': [random.randint(1, 365) for _ in range(n)],
        # time in minutes since midnight, between 6:00 (360) and 22:00 (1320)
        'time_minutes': [random.randint(360, 1320) for _ in range(n)],
        'longitude': [round(random.uniform(LON_MIN, LON_MAX), 6) for _ in range(n)],
        'latitude':  [round(random.uniform(LAT_MIN, LAT_MAX), 6) for _ in range(n)],
        'occupancy': [random.choice(occupancy_levels) for _ in range(n)]
    }
    return pd.DataFrame(data)

if __name__ == '__main__':
    # Generate 1,000 rows (adjust n as needed)
    df = generate_data(40000)
    df.to_csv('synthetic_occupancy_data.csv', index=False)
    print(df.head())
