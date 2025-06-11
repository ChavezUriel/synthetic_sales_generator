import numpy as np
from growth_functions import logarithmic_growth, linear_growth, sigmoid_growth

CONFIG = {
    'simulation': {
        'start_date': '2023-01-01',
        'end_date': '2026-12-31',
    },
    'initial_states': {
        'branches': 2,
        'agents_per_branch': 5,
        'customers': 50,
        'products': 30,
    },
    'growth_functions': {
        'branches': {
            'function': logarithmic_growth,
            'params': {'value_after_one_year': 3, 'growth_speed': 4, 'poly_noise_factor': 0.01}
        },
        'agents': {
            'function': sigmoid_growth,
            'params': {'max_value': 20, 'growth_rate': 0.01, 'pc_start': 0.1, 'poly_noise_factor': 0.01}
        },
        'customers': {
            'function': logarithmic_growth,
            'params': {'value_after_one_year': 50, 'growth_speed': 10, 'poly_noise_factor': 0.01}
        },
        'products': {
            'function': logarithmic_growth,
            'params': {'value_after_one_year': 10, 'growth_speed': 0.6, 'poly_noise_factor': 0.01}
        },
        'sales_frequency': {
            'function': logarithmic_growth,
            'params': {'value_after_one_year': 30, 'growth_speed': 5, 'poly_noise_factor': 0.01}
        },
        'price_multiplier': {
            'function': linear_growth,
            'params': {'initial': 1.0, 'rate_per_year': 0.05, 'poly_noise_factor': 0.01}
        },
        'customer_churn_lifespan': {
            'function': logarithmic_growth,
            'params': {'value_after_one_year': 365, 'growth_speed': 5, 'poly_noise_factor': 0.01}  # Avg lifespan grows to 1 year
        },
        'product_base_price_range': (5.0, 500.0),
        'transaction_quantity_range': (1, 10),
        'faker_locale': 'en_US',
    },
    'output': {
        'filename_csv': 'outputs/synthetic_sales_data.csv',
        'filename_parquet': 'outputs/synthetic_sales_data.parquet',
    }
}
