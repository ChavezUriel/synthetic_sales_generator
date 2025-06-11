import numpy as np
from growth_functions import logarithmic_growth, linear_growth, sigmoid_growth

CONFIG = {
    'simulation': {
        'start_date': '2020-01-01',
        'end_date': '2025-12-31',
    },
    'initial_states': {
        'branches': 3,
        'agents_per_branch': 3,
        'customers': 30,
        'products': 15,
    },
    'growth_functions': {
        'branches': {
            'function': sigmoid_growth,
            'params': {'max_value': 10, 'growth_rate': 0.02, 'pc_start': 0.2, 'poly_noise_factor': 0.03}
        },
        'agents': {
            'function': linear_growth,
            'params': {'initial': 9, 'rate_per_year': 3, 'poly_noise_factor': 0.03}
        },
        'customers': {
            'function': sigmoid_growth,
            'params': {'max_value': 100, 'growth_rate': 0.04, 'pc_start': 0.1, 'poly_noise_factor': 0.03}
        },
        'products': {
            'function': logarithmic_growth,
            'params': {'value_after_one_year': 20, 'growth_speed': 2, 'poly_noise_factor': 0.03}
        },
        'sales_frequency': {
            'function': logarithmic_growth,
            'params': {'value_after_one_year': 15, 'growth_speed': 1, 'poly_noise_factor': 0.03}
        },
        'price_multiplier': {
            'function': linear_growth,
            'params': {'initial': 1.0, 'rate_per_year': 0.02, 'poly_noise_factor': 0.03}
        },
        'customer_churn_lifespan': {
            'function': logarithmic_growth,
            'params': {'value_after_one_year': 180, 'growth_speed': 2, 'poly_noise_factor': 0.03}  # Avg lifespan grows to 6 months
        },
        'product_base_price_range': (2.0, 200.0),
        'transaction_quantity_range': (1, 5),
        'faker_locale': 'es_ES',
    },
    'output': {
        'filename_csv': 'outputs/synthetic_sales_data3.csv',
        'filename_parquet': 'outputs/synthetic_sales_data3.parquet',
    }
}
