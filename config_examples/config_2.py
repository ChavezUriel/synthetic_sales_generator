import numpy as np
from growth_functions import logarithmic_growth, linear_growth, sigmoid_growth

CONFIG = {
    'simulation': {
        'start_date': '2022-01-01',
        'end_date': '2027-12-31',
    },
    'initial_states': {
        'branches': 5,
        'agents_per_branch': 10,
        'customers': 200,
        'products': 100,
    },
    'growth_functions': {
        'branches': {
            'function': linear_growth,
            'params': {'initial': 5, 'rate_per_year': 2, 'poly_noise_factor': 0.02}
        },
        'agents': {
            'function': sigmoid_growth,
            'params': {'max_value': 100, 'growth_rate': 0.03, 'pc_start': 0.05, 'poly_noise_factor': 0.02}
        },
        'customers': {
            'function': logarithmic_growth,
            'params': {'value_after_one_year': 300, 'growth_speed': 8, 'poly_noise_factor': 0.02}
        },
        'products': {
            'function': linear_growth,
            'params': {'initial': 100, 'rate_per_year': 20, 'poly_noise_factor': 0.02}
        },
        'sales_frequency': {
            'function': linear_growth,
            'params': {'initial': 50, 'rate_per_year': 30, 'poly_noise_factor': 0.02}
        },
        'price_multiplier': {
            'function': linear_growth,
            'params': {'initial': 1.0, 'rate_per_year': 0.1, 'poly_noise_factor': 0.02}
        },
        'customer_churn_lifespan': {
            'function': sigmoid_growth,
            'params': {'max_value': 730, 'growth_rate': 0.01, 'pc_start': 0.1, 'poly_noise_factor': 0.02}  # Avg lifespan up to 2 years
        },
        'product_base_price_range': (10.0, 1000.0),
        'transaction_quantity_range': (1, 20),
        'faker_locale': 'fr_FR',
    },
    'output': {
        'filename_csv': 'outputs/synthetic_sales_data2.csv',
        'filename_parquet': 'outputs/synthetic_sales_data2.parquet',
    }
}
