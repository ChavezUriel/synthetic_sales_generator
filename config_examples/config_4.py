import numpy as np
from growth_functions import logarithmic_growth, linear_growth, sigmoid_growth

CONFIG = {
    'simulation': {
        'start_date': '2021-06-01',
        'end_date': '2024-06-01',
    },
    'initial_states': {
        'branches': 1,
        'agents_per_branch': 2,
        'customers': 10,
        'products': 5,
    },
    'growth_functions': {
        'branches': {
            'function': linear_growth,
            'params': {'initial': 1, 'rate_per_year': 1, 'poly_noise_factor': 0.05}
        },
        'agents': {
            'function': linear_growth,
            'params': {'initial': 2, 'rate_per_year': 2, 'poly_noise_factor': 0.05}
        },
        'customers': {
            'function': linear_growth,
            'params': {'initial': 10, 'rate_per_year': 10, 'poly_noise_factor': 0.05}
        },
        'products': {
            'function': linear_growth,
            'params': {'initial': 5, 'rate_per_year': 2, 'poly_noise_factor': 0.05}
        },
        'sales_frequency': {
            'function': linear_growth,
            'params': {'initial': 5, 'rate_per_year': 3, 'poly_noise_factor': 0.05}
        },
        'price_multiplier': {
            'function': linear_growth,
            'params': {'initial': 1.0, 'rate_per_year': 0.01, 'poly_noise_factor': 0.05}
        },
        'customer_churn_lifespan': {
            'function': linear_growth,
            'params': {'initial': 60, 'rate_per_year': 30, 'poly_noise_factor': 0.05}  # Avg lifespan starts at 2 months, grows by 1 month/year
        },
        'product_base_price_range': (1.0, 50.0),
        'transaction_quantity_range': (1, 2),
        'faker_locale': 'de_DE',
    },
    'output': {
        'filename_csv': 'outputs/synthetic_sales_data4.csv',
        'filename_parquet': 'outputs/synthetic_sales_data4.parquet',
    }
}
