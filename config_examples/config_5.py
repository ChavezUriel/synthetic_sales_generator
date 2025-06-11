import numpy as np
from growth_functions import logarithmic_growth, linear_growth, sigmoid_growth

CONFIG = {
    'simulation': {
        'start_date': '2024-01-01',
        'end_date': '2029-01-01',
    },
    'initial_states': {
        'branches': 10,
        'agents_per_branch': 20,
        'customers': 1000,
        'products': 200,
    },
    'growth_functions': {
        'branches': {
            'function': sigmoid_growth,
            'params': {'max_value': 50, 'growth_rate': 0.02, 'pc_start': 0.05, 'poly_noise_factor': 0.02}
        },
        'agents': {
            'function': sigmoid_growth,
            'params': {'max_value': 400, 'growth_rate': 0.03, 'pc_start': 0.1, 'poly_noise_factor': 0.02}
        },
        'customers': {
            'function': sigmoid_growth,
            'params': {'max_value': 5000, 'growth_rate': 0.04, 'pc_start': 0.1, 'poly_noise_factor': 0.02}
        },
        'products': {
            'function': sigmoid_growth,
            'params': {'max_value': 500, 'growth_rate': 0.01, 'pc_start': 0.05, 'poly_noise_factor': 0.02}
        },
        'sales_frequency': {
            'function': sigmoid_growth,
            'params': {'max_value': 1000, 'growth_rate': 0.05, 'pc_start': 0.1, 'poly_noise_factor': 0.02}
        },
        'price_multiplier': {
            'function': linear_growth,
            'params': {'initial': 1.0, 'rate_per_year': 0.2, 'poly_noise_factor': 0.02}
        },
        'customer_churn_lifespan': {
            'function': sigmoid_growth,
            'params': {'max_value': 1095, 'growth_rate': 0.01, 'pc_start': 0.05, 'poly_noise_factor': 0.02}  # Avg lifespan up to 3 years
        },
        'product_base_price_range': (20.0, 2000.0),
        'transaction_quantity_range': (1, 50),
        'faker_locale': 'it_IT',
    },
    'output': {
        'filename_csv': 'outputs/synthetic_sales_data5.csv',
        'filename_parquet': 'outputs/synthetic_sales_data5.parquet',
    }
}
