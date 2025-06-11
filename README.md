# Synthetic Relational Sales Data Generator

This repository provides a highly configurable synthetic sales data generator for simulating realistic, time-evolving sales transactions across branches, agents, customers, and products.

## Features

- **Flexible growth functions**: Logarithmic, sigmoid, linear, and more, with support for Gaussian and polynomial noise.
- **Customer churn modeling**: Configurable customer lifespan and churn mechanism.
- **Batch configuration**: Easily run multiple scenarios using example configs in `config_examples/`.
- **Comprehensive reporting**: Generates detailed plots and summary statistics (see `synthetic_data_report.png`).
- **Output in CSV and Parquet**: All generated data is saved in the `outputs/` directory.
- **BigQuery integration**: Utilities in `upload/` (not tracked by git) to upload generated data directly to Google BigQuery.

## Usage

### Generate Synthetic Data

Run the main script to generate data using all configs in `config_examples/`:

```sh
python main.py
```

This will generate synthetic sales data and reports for each configuration.

## Configuration

- Main configuration: `config.py`
- Example configurations: `config_examples/config_*.py`
- Growth functions: `growth_functions.py`
- Data generator: `synthetic_data_generator.py`

## Requirements

- Python 3.8+
- pandas, numpy, matplotlib, pyarrow, faker, google-cloud-bigquery

Install requirements:

```sh
pip install -r requirements.txt
```

## Example Plots

See `synthetic_data_report.png` for sample visualizations, including:
- Transactions per day
- Sales by branch, agent, product, customer
- Distribution of quantities and prices
- Cumulative and monthly sales
- Customer activity and churn

## License

MIT License

---

*Ideal for data science, analytics, and machine learning prototyping where realistic, relational sales data is needed.*
