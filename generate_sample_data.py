import pandas as pd
import numpy as np

# Define crop characteristics (ranges for N, P, K, temperature, humidity, ph, rainfall)
crop_ranges = {
    'rice': {
        'N': (50, 100), 'P': (30, 70), 'K': (40, 80),
        'temperature': (20, 30), 'humidity': (60, 90), 'ph': (5.5, 7.0), 'rainfall': (150, 300)
    },
    'wheat': {
        'N': (70, 120), 'P': (40, 80), 'K': (30, 70),
        'temperature': (10, 25), 'humidity': (40, 70), 'ph': (6.0, 7.5), 'rainfall': (50, 150)
    },
    'maize': {
        'N': (60, 110), 'P': (50, 90), 'K': (40, 80),
        'temperature': (18, 32), 'humidity': (50, 80), 'ph': (5.8, 7.2), 'rainfall': (100, 200)
    },
    # Add more crops as needed
    'cotton': {
        'N': (40, 90), 'P': (30, 70), 'K': (50, 90),
        'temperature': (25, 35), 'humidity': (40, 70), 'ph': (5.5, 7.5), 'rainfall': (50, 100)
    },
    'sugarcane': {
        'N': (80, 130), 'P': (40, 80), 'K': (50, 100),
        'temperature': (20, 35), 'humidity': (60, 85), 'ph': (6.0, 7.5), 'rainfall': (100, 250)
    }
}

def generate_crop_data(crop, n_samples=200):
    data = []
    for _ in range(n_samples):
        sample = {}
        for param, (min_val, max_val) in crop_ranges[crop].items():
            if param in ['ph']:
                # For pH, use a normal distribution around the middle of the range
                mean = (min_val + max_val) / 2
                std = (max_val - min_val) / 6  # 99.7% within range
                sample[param] = np.random.normal(mean, std)
            else:
                # For other parameters, use uniform distribution
                sample[param] = np.random.uniform(min_val, max_val)
        sample['label'] = crop
        data.append(sample)
    return pd.DataFrame(data)

# Generate data for all crops
dfs = []
for crop in crop_ranges.keys():
    dfs.append(generate_crop_data(crop))

# Combine all data
all_data = pd.concat(dfs, ignore_index=True)

# Shuffle the data
all_data = all_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to CSV
all_data.to_csv('soil_crop_recommendation.csv', index=False)
print("Sample dataset generated successfully with", len(all_data), "samples")
