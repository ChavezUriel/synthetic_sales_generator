# %%
# main.py
import pandas as pd
import matplotlib.pyplot as plt
from config import CONFIG
from synthetic_data_generator import SyntheticSalesGenerator
import os
import importlib.util

def main():
    """
    Main function to run the synthetic data generation process.
    """
    print("=========================================")
    print(" Synthetic Relational Sales Data Generator ")
    print("=========================================")

    # 1. Instantiate the generator with the configuration
    generator = SyntheticSalesGenerator(CONFIG)

    # 2. Run the simulation
    generator.run_simulation()

    # 3. Get the final consolidated and validated data
    # Set add_names=False to get raw data with IDs only
    final_sales_df = generator.get_final_data(add_names=True)

    # 4. Display some information about the generated data
    print("\n--- Generated Data Summary ---")
    if not final_sales_df.empty:
        print(f"Shape: {final_sales_df.shape}")
        print(f"Date Range: {final_sales_df['OrderDate'].min().date()} to {final_sales_df['OrderDate'].max().date()}")
        print("Columns:", final_sales_df.columns.tolist())
        print("Data Head:")
        print(final_sales_df.head())

        # --- Optional: Display counts from final merged data ---
        print("\n--- Entity Counts in Final Transactions ---")
        print(f"Unique Branches: {final_sales_df['BranchID'].nunique()}")
        print(f"Unique Sales Agents: {final_sales_df['AgentID'].nunique()}")
        print(f"Unique Customers: {final_sales_df['CustomerID'].nunique()}")
        print(f"Unique Products: {final_sales_df['ProductID'].nunique()}")

        # 5. Save the data
        generator.save_data(final_sales_df)

        # 6. Generate plots and report
        generate_report(final_sales_df)
    else:
        print("No data was generated.")

    print("\nProcess finished.")
    print("=========================================")

def run_all_configs():
    """
    Run data generation for all config files in config_examples directory.
    Each config is loaded and used to instantiate the generator and run the pipeline.
    """
    config_dir = os.path.join(os.path.dirname(__file__), 'config_examples')
    config_files = [f for f in os.listdir(config_dir) if f.startswith('config_') and f.endswith('.py')]
    for config_file in config_files:
        config_path = os.path.join(config_dir, config_file)
        spec = importlib.util.spec_from_file_location('config_module', config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        config = config_module.CONFIG
        print(f"\n==============================")
        print(f"Running data generation for: {config_file}")
        print(f"==============================")
        generator = SyntheticSalesGenerator(config)
        generator.run_simulation()
        final_sales_df = generator.get_final_data(add_names=True)
        if not final_sales_df.empty:
            print(f"Shape: {final_sales_df.shape}")
            print(f"Date Range: {final_sales_df['OrderDate'].min().date()} to {final_sales_df['OrderDate'].max().date()}")
            print("Columns:", final_sales_df.columns.tolist())
            print("Data Head:")
            print(final_sales_df.head())
            print("\n--- Entity Counts in Final Transactions ---")
            print(f"Unique Branches: {final_sales_df['BranchID'].nunique()}")
            print(f"Unique Sales Agents: {final_sales_df['AgentID'].nunique()}")
            print(f"Unique Customers: {final_sales_df['CustomerID'].nunique()}")
            print(f"Unique Products: {final_sales_df['ProductID'].nunique()}")
            generator.save_data(final_sales_df)
            generate_report(final_sales_df)
        else:
            print("No data was generated.")
        print("\nProcess finished for", config_file)
        print("===========================================\n")

def generate_report(df):
    """
    Generate a report with at least 10 different plots visualizing the generated data.
    """
    print("\nGenerating report with visualizations...")
    plt.style.use('seaborn-v0_8-darkgrid')

    print("TRY: \n\n", df.columns)
    # print("TRY: \n\n", df.groupby('BranchID')['Sales'])
    
    plots = [
        # 1. Transactions over time
        {
            'title': 'Transactions per Day',
            'plot': lambda ax: df.groupby('OrderDate').size().plot(ax=ax, title='Transactions per Day')
        },
        # 2. Sales by Branch
        {
            'title': 'Total Sales by Branch',
            'plot': lambda ax: df.groupby('BranchID')['Sales'].sum().plot.bar(ax=ax, title='Total Sales by Branch')
        },
        # 3. Sales by Agent
        {
            'title': 'Total Sales by Agent',
            'plot': lambda ax: df.groupby('AgentID')['Sales'].sum().plot.bar(ax=ax, title='Total Sales by Agent')
        },
        # 4. Sales by Product
        {
            'title': 'Total Sales by Product',
            'plot': lambda ax: df.groupby('ProductID')['Sales'].sum().plot.bar(ax=ax, title='Total Sales by Product')
        },
        # 5. Sales by Customer
        {
            'title': 'Total Sales by Customer',
            'plot': lambda ax: df.groupby('CustomerID')['Sales'].sum().plot(ax=ax, title='Total Sales by Customer')
        },
        # 6. Distribution of Transaction Quantities
        {
            'title': 'Distribution of Transaction Quantities',
            'plot': lambda ax: df['Quantity'].plot.hist(ax=ax, bins=20, title='Distribution of Transaction Quantities')
        },
        # 7. Distribution of Product Prices
        {
            'title': 'Distribution of Product Prices',
            'plot': lambda ax: df['UnitPrice'].plot.hist(ax=ax, bins=20, title='Distribution of Product Prices')
        },
        # 8. Average Sales per Day
        {
            'title': 'Average Sales per Day',
            'plot': lambda ax: df.groupby('OrderDate')['Sales'].mean().plot(ax=ax, title='Average Sales per Day')
        },
        # 9. Cumulative Sales Over Time
        {
            'title': 'Cumulative Sales Over Time',
            'plot': lambda ax: df.groupby('OrderDate')['Sales'].sum().cumsum().plot(ax=ax, title='Cumulative Sales Over Time')
        },
        # 10. Number of Unique Customers per Day
        {
            'title': 'Unique Customers per Day',
            'plot': lambda ax: df.groupby('OrderDate')['CustomerID'].nunique().plot(ax=ax, title='Unique Customers per Day')
        },
        # 11. Active Customers per Month
        {
            'title': 'Active Customers per Month',
            'plot': lambda ax: df.groupby(df['OrderDate'].dt.to_period('M'))['CustomerID'].nunique().plot(ax=ax, title='Active Customers per Month')
        },
        # 12. Total Sales per Month
        {
            'title': 'Total Sales per Month',
            'plot': lambda ax: df.groupby(df['OrderDate'].dt.to_period('M'))['Sales'].sum().plot(ax=ax, title='Total Sales per Month')
        },
    ]
    fig, axes = plt.subplots(6, 2, figsize=(22, 28))
    axes = axes.flatten()
    for i, plot_info in enumerate(plots):
        ax = axes[i]
        plot_info['plot'](ax)
        ax.set_title(plot_info['title'])
        ax.set_xlabel('')
        ax.set_ylabel('')
    plt.tight_layout()
    plt.savefig('synthetic_data_report.png')
    # plt.show()
    print("Report saved as 'synthetic_data_report.png'.")

if __name__ == "__main__":
    # Optional: Install required libraries if needed
    # !pip install pandas faker numpy pyarrow # Uncomment if running in an environment where installation is needed
    run_all_configs()

# %%
