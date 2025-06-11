# config.py
"""
Configuration settings for the synthetic sales data generator.
"""

import numpy as np
from growth_functions import constant_growth, linear_growth, exponential_growth, constant_value, logarithmic_growth, sigmoid_growth

CONFIG = {
    'simulation': {
        'start_date': '2018-01-01',
        'end_date': '2027-12-31', # Shorter period for quicker testing
    },
    'initial_states': {
        'branches': 2,
        'agents_per_branch': 5,
        'customers': 50,
        'products': 30,
    },
}

# Link growth function initial values to initial_states
CONFIG['growth_functions'] = {
    # --- Quantity Growth ---
    # Target number of entities over time (t = days since start)
    'branches': {
        'function': logarithmic_growth,
        'params': {'value_after_one_year': 3, "growth_speed": 4, 'poly_noise_factor': 0.01} # Value after 1 year (365 days)
    },
    'agents': { # Target total agents
        'function': sigmoid_growth,
        'params': {'max_value': 20, 'growth_rate': 0.01, 'pc_start': 0.1, 'poly_noise_factor': 0.01} # Example sigmoid: up to 20 agents, slow growth
    },
    'customers': {
        'function': logarithmic_growth,
        'params': {'value_after_one_year': 50, "growth_speed": 10, 'poly_noise_factor': 0.01} # Example: 200 customers after 1 year
    },
    'products': {
        'function': logarithmic_growth,
        'params': {'value_after_one_year': 10, "growth_speed": 0.6, 'poly_noise_factor': 0.01} # Example: 40 products after 1 year
    },
    # --- Rate/Value Growth ---
    'sales_frequency': { # Target transactions per day
        'function': logarithmic_growth,
        'params': {'value_after_one_year': 30,  "growth_speed": 5, 'poly_noise_factor': 0.01} # Example: 120 sales/day after 1 year
    },
    'price_multiplier': { # Multiplier applied to BaseUnitPrice
        'function': linear_growth,
        'params': {'initial': 1.0, 'rate_per_year': 0.05, 'poly_noise_factor': 0.01} # 5% price increase per year
    },
    # --- Other Parameters ---
    'product_base_price_range': (5.0, 500.0),
    'transaction_quantity_range': (1, 10),
    'faker_locale': 'en_US',
    # Optional: Add PhaseOutDate/EndDate simulation params here if implemented
}

CONFIG['output'] = {
    'filename_csv': 'synthetic_sales_data.csv',
    'filename_parquet': 'synthetic_sales_data.parquet', # Optional
}