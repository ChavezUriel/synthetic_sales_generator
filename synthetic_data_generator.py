# synthetic_data_generator.py
import pandas as pd
import numpy as np
from faker import Faker
import datetime
import time
import random

class SyntheticSalesGenerator:
    """
    Generates synthetic relational sales data based on configurable
    time-dependent growth parameters for entities (Branches, Agents,
    Customers, Products) and transaction characteristics.
    """
    def __init__(self, config):
        """Initializes the generator with configuration."""
        self.config = config
        self.start_date = pd.to_datetime(config['simulation']['start_date'])
        self.end_date = pd.to_datetime(config['simulation']['end_date'])
        self.date_range = pd.date_range(self.start_date, self.end_date, freq='D')
        self.fake = Faker(config['growth_functions'].get('faker_locale', 'en_US'))

        # Initialize entity counters
        self._next_branch_id = 1
        self._next_agent_id = 1
        self._next_customer_id = 1
        self._next_product_id = 1

        # Initialize empty DataFrames
        self._initialize_entities()

        self.daily_transactions_list = []
        # Store current prices {ProductID: CurrentUnitPrice}
        self.current_prices = self._get_initial_prices()

        print("Generator Initialized.")
        print(f"Simulation Period: {self.start_date.date()} to {self.end_date.date()}")


    def _initialize_entities(self):
        """Creates initial DataFrames based on config['initial_states']."""
        print("Initializing entities...")
        # Branches
        initial_branches = []
        for _ in range(self.config['initial_states']['branches']):
            initial_branches.append({
                'BranchID': self._next_branch_id,
                'BranchName': self.fake.company() + " Branch",
                'OpeningDate': self.start_date,
                'EndDate': pd.NaT # NaT represents Not a Time (null for dates)
            })
            self._next_branch_id += 1
        self.branches_df = pd.DataFrame(initial_branches)
        self.branches_df['OpeningDate'] = pd.to_datetime(self.branches_df['OpeningDate'])
        self.branches_df['EndDate'] = pd.to_datetime(self.branches_df['EndDate'])

        # Agents
        initial_agents = []
        if not self.branches_df.empty:
            agents_per_branch = self.config['initial_states']['agents_per_branch']
            initial_branch_ids = self.branches_df['BranchID'].tolist()
            for branch_id in initial_branch_ids:
                for _ in range(agents_per_branch):
                    initial_agents.append({
                        'AgentID': self._next_agent_id,
                        'AgentName': self.fake.name(),
                        'AssignedBranchID': branch_id,
                        'StartDate': self.start_date,
                        'EndDate': pd.NaT
                    })
                    self._next_agent_id += 1
        self.agents_df = pd.DataFrame(initial_agents)
        if not self.agents_df.empty:
             self.agents_df['StartDate'] = pd.to_datetime(self.agents_df['StartDate'])
             self.agents_df['EndDate'] = pd.to_datetime(self.agents_df['EndDate'])


        # Customers
        initial_customers = []
        churn_func_config = self.config['growth_functions'].get('customer_churn_lifespan', None)
        for _ in range(self.config['initial_states']['customers']):
            acquisition_date = self.start_date
            churn_date = pd.NaT
            if churn_func_config:
                churn_func = churn_func_config.get('function', None)
                churn_params = churn_func_config.get('params', {})
                if churn_func:
                    lifespan_days = int(np.round(churn_func(0, churn_params)))
                    if lifespan_days > 0:
                        churn_date = acquisition_date + pd.Timedelta(days=lifespan_days)
            initial_customers.append({
                'CustomerID': self._next_customer_id,
                'CustomerName': self.fake.name(),
                'AcquisitionDate': acquisition_date, # Assume initial customers acquired on start date
                'ChurnDate': churn_date
            })
            self._next_customer_id += 1
        self.customers_df = pd.DataFrame(initial_customers)
        self.customers_df['AcquisitionDate'] = pd.to_datetime(self.customers_df['AcquisitionDate'])
        self.customers_df['ChurnDate'] = pd.to_datetime(self.customers_df['ChurnDate'])


        # Products
        initial_products = []
        min_price, max_price = self.config['growth_functions']['product_base_price_range']
        categories = ['Electronics', 'Apparel', 'Home Goods', 'Groceries', 'Toys', 'Books']
        for _ in range(self.config['initial_states']['products']):
            initial_products.append({
                'ProductID': self._next_product_id,
                'ProductName': self.fake.word().capitalize() + " " + self.fake.word().capitalize(),
                'Category': self.fake.random_element(elements=categories),
                'IntroductionDate': self.start_date,
                'BaseUnitPrice': np.round(self.fake.random_number(digits=5) / 10000 * (max_price - min_price) + min_price, 2),
                'PhaseOutDate': pd.NaT
            })
            self._next_product_id += 1
        self.products_df = pd.DataFrame(initial_products)
        self.products_df['IntroductionDate'] = pd.to_datetime(self.products_df['IntroductionDate'])
        self.products_df['PhaseOutDate'] = pd.to_datetime(self.products_df['PhaseOutDate'])

        print(f"Initial Branches: {len(self.branches_df)}")
        print(f"Initial Agents: {len(self.agents_df)}")
        print(f"Initial Customers: {len(self.customers_df)}")
        print(f"Initial Products: {len(self.products_df)}")


    def _get_initial_prices(self):
        """Get base prices for initially active products."""
        initial_prices = {}
        if not self.products_df.empty:
             active_products = self.products_df[self.products_df['IntroductionDate'] <= self.start_date]
             initial_prices = pd.Series(active_products.BaseUnitPrice.values, index=active_products.ProductID).to_dict()
        return initial_prices


    def _calculate_target(self, entity_key, current_date):
        """Calculates the target value using the configured growth function."""
        t = (current_date - self.start_date).days
        func_config = self.config['growth_functions'].get(entity_key, {})
        func = func_config.get('function', None)
        params = func_config.get('params', {})

        if func:
            return func(t, params)
        else:
            # Default: return initial value if no function defined
            if entity_key == 'branches': return self.config['initial_states']['branches']
            if entity_key == 'agents': return self.config['initial_states']['branches'] * self.config['initial_states']['agents_per_branch']
            if entity_key == 'customers': return self.config['initial_states']['customers']
            if entity_key == 'products': return self.config['initial_states']['products']
            if entity_key == 'price_multiplier': return 1.0 # Default multiplier
            if entity_key == 'sales_frequency': return 100 # Default transactions
            return 0 # Default for unknown keys


    def _get_active_df(self, df, date_col_start, date_col_end, current_date):
        """Filters DataFrame for records active on current_date."""
        if df.empty:
            return df

        active_filter = (df[date_col_start] <= current_date) & \
                        (pd.isna(df[date_col_end]) | (df[date_col_end] > current_date))
        return df[active_filter]


    def _apply_growth_and_add_entities(self, current_date):
        """Calculates targets and adds new entities to DataFrames."""
        new_entities = {'branches': [], 'agents': [], 'customers': [], 'products': []}

        # --- Branches ---
        target_branch_count = int(np.round(self._calculate_target('branches', current_date)))
        current_active_branches = self._get_active_df(self.branches_df, 'OpeningDate', 'EndDate', current_date)
        num_new_branches = max(0, target_branch_count - len(current_active_branches))

        for _ in range(num_new_branches):
            new_branch = {
                'BranchID': self._next_branch_id,
                'BranchName': self.fake.company() + " Branch",
                'OpeningDate': current_date,
                'EndDate': pd.NaT
            }
            new_entities['branches'].append(new_branch)
            self._next_branch_id += 1

        # --- Agents ---
        target_agent_count = int(np.round(self._calculate_target('agents', current_date)))
        current_active_agents = self._get_active_df(self.agents_df, 'StartDate', 'EndDate', current_date)
        num_new_agents = max(0, target_agent_count - len(current_active_agents))

        # Get branches active *today* for assignment
        assignable_branches = self._get_active_df(self.branches_df, 'OpeningDate', 'EndDate', current_date)
        if not assignable_branches.empty:
            assignable_branch_ids = assignable_branches['BranchID'].tolist()
            for _ in range(num_new_agents):
                 # Assign to a randomly selected *currently active* branch
                assigned_branch_id = self.fake.random_element(elements=assignable_branch_ids)
                new_agent = {
                    'AgentID': self._next_agent_id,
                    'AgentName': self.fake.name(),
                    'AssignedBranchID': assigned_branch_id,
                    'StartDate': current_date,
                    'EndDate': pd.NaT
                }
                new_entities['agents'].append(new_agent)
                self._next_agent_id += 1
        elif num_new_agents > 0:
             print(f"Warning ({current_date.date()}): Cannot add {num_new_agents} new agents, no active branches found.")


        # --- Customers ---
        target_customer_count = int(np.round(self._calculate_target('customers', current_date)))
        current_active_customers = self._get_active_df(self.customers_df, 'AcquisitionDate', 'ChurnDate', current_date)
        num_new_customers = max(0, target_customer_count - len(current_active_customers))
        churn_func_config = self.config['growth_functions'].get('customer_churn_lifespan', None)
        for _ in range(num_new_customers):
            churn_date = pd.NaT
            if churn_func_config:
                churn_func = churn_func_config.get('function', None)
                churn_params = churn_func_config.get('params', {})
                if churn_func:
                    lifespan_days = int(np.round(churn_func(0, churn_params)))
                    if lifespan_days > 0:
                        churn_date = current_date + pd.Timedelta(days=lifespan_days)
            new_customer = {
                'CustomerID': self._next_customer_id,
                'CustomerName': self.fake.name(),
                'AcquisitionDate': current_date,
                'ChurnDate': churn_date
            }
            new_entities['customers'].append(new_customer)
            self._next_customer_id += 1


        # --- Products ---
        target_product_count = int(np.round(self._calculate_target('products', current_date)))
        current_active_products = self._get_active_df(self.products_df, 'IntroductionDate', 'PhaseOutDate', current_date)
        num_new_products = max(0, target_product_count - len(current_active_products))
        min_price, max_price = self.config['growth_functions']['product_base_price_range']
        categories = self.products_df['Category'].unique().tolist() if not self.products_df.empty else ['Default']

        for _ in range(num_new_products):
            base_price = np.round(self.fake.random_number(digits=5) / 10000 * (max_price - min_price) + min_price, 2)
            new_prod = {
                'ProductID': self._next_product_id,
                'ProductName': self.fake.word().capitalize() + " " + self.fake.word().capitalize(),
                'Category': self.fake.random_element(elements=categories),
                'IntroductionDate': current_date,
                'BaseUnitPrice': base_price,
                'PhaseOutDate': pd.NaT
            }
            new_entities['products'].append(new_prod)
            # Add new product's initial price to current prices
            self.current_prices[self._next_product_id] = base_price
            self._next_product_id += 1


        # --- Concatenate new entities ---
        if new_entities['branches']:
            self.branches_df = pd.concat([self.branches_df, pd.DataFrame(new_entities['branches'])], ignore_index=True)
        if new_entities['agents']:
            self.agents_df = pd.concat([self.agents_df, pd.DataFrame(new_entities['agents'])], ignore_index=True)
            # Ensure date columns are datetime type after concat
            self.agents_df['StartDate'] = pd.to_datetime(self.agents_df['StartDate'])
            self.agents_df['EndDate'] = pd.to_datetime(self.agents_df['EndDate'])
        if new_entities['customers']:
            self.customers_df = pd.concat([self.customers_df, pd.DataFrame(new_entities['customers'])], ignore_index=True)
            self.customers_df['AcquisitionDate'] = pd.to_datetime(self.customers_df['AcquisitionDate'])
            self.customers_df['ChurnDate'] = pd.to_datetime(self.customers_df['ChurnDate'])
        if new_entities['products']:
            self.products_df = pd.concat([self.products_df, pd.DataFrame(new_entities['products'])], ignore_index=True)
            self.products_df['IntroductionDate'] = pd.to_datetime(self.products_df['IntroductionDate'])
            self.products_df['PhaseOutDate'] = pd.to_datetime(self.products_df['PhaseOutDate'])


    def _update_prices(self, current_date):
        """Applies price growth function to active products."""
        price_multiplier = self._calculate_target('price_multiplier', current_date)

        # Get products active today
        active_prods = self._get_active_df(self.products_df, 'IntroductionDate', 'PhaseOutDate', current_date)

        if not active_prods.empty:
             # Update current_prices dictionary for active products
             # Price = BasePrice * Multiplier(t)
             for index, product in active_prods.iterrows():
                 prod_id = product['ProductID']
                 base_price = product['BaseUnitPrice']
                 self.current_prices[prod_id] = np.round(base_price * price_multiplier, 2)


    def _generate_daily_transactions(self, current_date):
        """Generates transaction records for the current day."""
        daily_tx_list = []
        target_transactions = self._calculate_target('sales_frequency', current_date)
        # Add randomness: draw from Poisson distribution around the target mean
        num_transactions = np.random.poisson(max(0, target_transactions)) # Ensure non-negative

        if num_transactions == 0:
            return daily_tx_list # No transactions today

        # --- Filter active entities ONCE for the day ---
        active_branches = self._get_active_df(self.branches_df, 'OpeningDate', 'EndDate', current_date)
        active_agents = self._get_active_df(self.agents_df, 'StartDate', 'EndDate', current_date)
        active_customers = self._get_active_df(self.customers_df, 'AcquisitionDate', 'ChurnDate', current_date)
        active_products = self._get_active_df(self.products_df, 'IntroductionDate', 'PhaseOutDate', current_date)

        # --- Pre-filter agents by active branches ---
        if active_branches.empty or active_agents.empty or active_customers.empty or active_products.empty:
            # If any essential group is empty, cannot generate transactions
            return daily_tx_list

        active_branch_ids = active_branches['BranchID'].tolist()
        # Ensure agents are linked to currently active branches
        agents_in_active_branches = active_agents[active_agents['AssignedBranchID'].isin(active_branch_ids)]

        if agents_in_active_branches.empty:
             # No agents available in the currently active branches
            return daily_tx_list

        # --- Pre-fetch active product IDs and Customers for faster selection ---
        active_customer_ids = active_customers['CustomerID'].tolist()
        active_product_ids = active_products['ProductID'].tolist()
        min_qty, max_qty = self.config['growth_functions']['transaction_quantity_range']


        # --- Generate Transactions ---
        for _ in range(int(num_transactions)):
            try:
                # 1. Select Branch randomly from active branches
                selected_branch_id = random.choice(active_branch_ids)

                # 2. Select Agent assigned to the selected branch
                possible_agents = agents_in_active_branches[agents_in_active_branches['AssignedBranchID'] == selected_branch_id]
                if possible_agents.empty:
                    continue # Skip if no agent found for this branch today (should be rare if growth logic is sound)
                selected_agent_id = random.choice(possible_agents['AgentID'].tolist())

                # 3. Select Customer randomly
                selected_customer_id = random.choice(active_customer_ids)

                # 4. Select Product randomly
                selected_product_id = random.choice(active_product_ids)

                # 5. Assign Quantity
                quantity = random.randint(min_qty, max_qty)

                # 6. Determine Unit Price (using the updated current_prices dict)
                unit_price = np.float64(self.current_prices.get(selected_product_id, 0.0)) # Default to 0 if somehow missing
                if unit_price <= 0:
                    # print(f"Warning: Product {selected_product_id} has zero or negative price on {current_date.date()}")
                    continue # Skip transaction if price is invalid

                # 7. Calculate Total Sales
                sales = np.float64(np.round(quantity * unit_price, 2))

                # 8. Store Transaction Record (as dict for efficiency)
                transaction_record = {
                    'OrderDate': current_date,
                    'BranchID': selected_branch_id,
                    'AgentID': selected_agent_id,
                    'CustomerID': selected_customer_id,
                    'ProductID': selected_product_id,
                    'UnitPrice': unit_price,
                    'Quantity': quantity,
                    'Sales': sales
                }
                daily_tx_list.append(transaction_record)

            except IndexError:
                 # Handles potential errors if any list becomes empty unexpectedly during the loop
                 # (e.g., if active sets were somehow modified mid-loop, which shouldn't happen here)
                 print(f"Warning ({current_date.date()}): Could not select entities for a transaction. Check active sets.")
                 continue
            except Exception as e:
                print(f"Error during transaction generation on {current_date.date()}: {e}")
                continue


        return daily_tx_list


    def run_simulation(self):
        """Runs the main simulation loop."""
        print("\nStarting simulation...")
        start_time = time.time()
        total_days = len(self.date_range)

        for i, current_date in enumerate(self.date_range):
            # --- Daily Update Phase ---
            self._apply_growth_and_add_entities(current_date)
            self._update_prices(current_date)

            # --- Transaction Generation Phase ---
            daily_transactions = self._generate_daily_transactions(current_date)
            if daily_transactions: # Avoid extending empty lists
                self.daily_transactions_list.extend(daily_transactions)

            # Progress Indicator
            if (i + 1) % max(1, total_days // 10) == 0 or i == total_days - 1: # Print progress roughly 10 times
                 elapsed = time.time() - start_time
                 print(f"  Processed Day {i+1}/{total_days} ({current_date.date()}) - {len(self.daily_transactions_list)} total transactions generated. Elapsed: {elapsed:.2f}s")


        end_time = time.time()
        print(f"\nSimulation finished in {end_time - start_time:.2f} seconds.")
        print(f"Total transactions generated: {len(self.daily_transactions_list)}")


    def get_final_data(self, add_names=True):
        """Consolidates, optionally merges names, performs basic validation, and returns the final DataFrame."""
        print("Assembling final dataset...")
        if not self.daily_transactions_list:
            print("Warning: No transactions were generated.")
            # Define columns even for empty DataFrame to match schema
            cols = ['OrderDate', 'BranchID', 'AgentID', 'CustomerID', 'ProductID',
                     'UnitPrice', 'Quantity', 'Sales']
            if add_names:
                cols.extend(['Branch', 'Sales Agent', 'Customer', 'Product', 'Category'])
            return pd.DataFrame(columns=cols)

        # Create DataFrame from the list of dictionaries
        final_df = pd.DataFrame(self.daily_transactions_list)
        final_df['OrderDate'] = pd.to_datetime(final_df['OrderDate']) # Ensure correct dtype

        if add_names:
            print("  Merging entity names...")
            # Merge Branch Names
            final_df = pd.merge(final_df, self.branches_df[['BranchID', 'BranchName']], on='BranchID', how='left')
            # Merge Agent Names
            final_df = pd.merge(final_df, self.agents_df[['AgentID', 'AgentName']], on='AgentID', how='left')
            # Merge Customer Names
            final_df = pd.merge(final_df, self.customers_df[['CustomerID', 'CustomerName']], on='CustomerID', how='left')
            # Merge Product Names and Category
            final_df = pd.merge(final_df, self.products_df[['ProductID', 'ProductName', 'Category']], on='ProductID', how='left')

            # Rename columns for clarity
            final_df.rename(columns={
                'BranchName': 'Branch',
                'AgentName': 'Sales Agent',
                'CustomerName': 'Customer',
                'ProductName': 'Product'
                # Category is already named correctly
            }, inplace=True)

             # Reorder columns for presentation
            final_columns_order = [
                'OrderDate', 'Branch', 'Sales Agent', 'Customer', 'Product', 'Category',
                'UnitPrice', 'Quantity', 'Sales',
                'BranchID', 'AgentID', 'CustomerID', 'ProductID' # Keep IDs for reference
            ]
            # Filter to only include columns that actually exist (in case merging failed silently)
            final_columns_order = [col for col in final_columns_order if col in final_df.columns]
            final_df = final_df[final_columns_order]

        # --- Validation Checks ---
        print("  Performing basic validation checks...")
        # 1. Check for Missing Names (if merged)
        if add_names:
            missing_names = final_df[['Branch', 'Sales Agent', 'Customer', 'Product']].isnull().sum()
            if missing_names.sum() > 0:
                 print(f"Warning: Found missing names after merge:\n{missing_names[missing_names > 0]}")

        # 2. Financial Consistency Check (allowing for float precision issues)
        expected_sales = final_df['Quantity'] * final_df['UnitPrice']
        discrepancy = np.abs(final_df['Sales'] - expected_sales)
        # Check for discrepancies larger than a small tolerance (e.g., 0.01 for cents)
        large_discrepancies = final_df[discrepancy > 0.01]
        if not large_discrepancies.empty:
            print(f"Warning: Found {len(large_discrepancies)} rows where Sales != Quantity * UnitPrice (tolerance 0.01). Example discrepancy: {discrepancy.max():.4f}")

        # 3. Temporal Validity (Basic check: OrderDate should not be before entity start dates)
        # Merging start dates back might be slow; rely on generation logic correctness.
        # A sample check could be done if performance allows.

        print("Final dataset assembly complete.")
        return final_df

    def save_data(self, final_df):
        """Saves the final DataFrame to CSV and/or Parquet."""
        
        
        
        if final_df.empty:
            print("No data to save.")
            return

        csv_path = self.config['output'].get('filename_csv', None)
        parquet_path = self.config['output'].get('filename_parquet', None)

        if csv_path:
            try:
                final_df.to_csv(csv_path, index=False)
                print(f"Data saved to {csv_path}")
            except Exception as e:
                 print(f"Error saving to CSV {csv_path}: {e}")

        if parquet_path:
             try:
                 # Requires 'pyarrow' or 'fastparquet' library
                 # pip install pyarrow  OR pip install fastparquet
                 final_df.to_parquet(parquet_path, index=False)
                 print(f"Data saved to {parquet_path}")
             except ImportError:
                 print("Warning: Cannot save to Parquet. Please install 'pyarrow' or 'fastparquet'.")
                 print("  pip install pyarrow")
             except Exception as e:
                 print(f"Error saving to Parquet {parquet_path}: {e}")