import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_datasets(output_dir: str | Path) -> None:
    """
    Create sample datasets for testing the CSV query chatbot.
    Generates both small (8 rows) and large (50,000 rows) datasets.

    Args:
        output_dir: Directory to save the CSV files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set random seed for reproducibility
    np.random.seed(42)

    # Common column definitions
    products = ['Laptop', 'Smartphone', 'Tablet', 'Smartwatch', 'Headphones']
    regions = ['North', 'South', 'East', 'West', 'Central']

    def generate_data(num_rows: int) -> pd.DataFrame:
        """Generate a dataset with specified number of rows"""
        return pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=num_rows),
            'product': np.random.choice(products, num_rows),
            'region': np.random.choice(regions, num_rows),
            'sales_amount': np.random.uniform(100, 2000, num_rows).round(2),
            'units_sold': np.random.randint(1, 50, num_rows),
            'customer_satisfaction': np.random.uniform(3.0, 5.0, num_rows).round(1),
            'delivery_time_days': np.random.randint(1, 10, num_rows),
            'is_promotional': np.random.choice([True, False], num_rows)
        })

    # Generate small dataset (6 rows)
    small_df = generate_data(6)
    small_df.to_csv(output_dir / 'small_sales_data.csv', index=False)
    logger.info("Created small dataset with 6 rows")

    # Generate large dataset (20,000 rows)
    large_df = generate_data(20000)
    large_df.to_csv(output_dir / 'large_sales_data.csv', index=False)
    logger.info("Created large dataset with 20,000 rows")

def print_dataset_info(file_path: str | Path) -> None:
    """
    Print basic information about a dataset.

    Args:
        file_path: Path to the CSV file
    """
    df = pd.read_csv(file_path)
    print(f"\nDataset: {Path(file_path).name}")
    print("-" * 50)
    print(f"Shape: {df.shape}")
    print("\nSample data:")
    print(df.head(3))
    print("\nData types:")
    print(df.dtypes)
    print("\nSummary statistics:")
    print(df.describe())

if __name__ == "__main__":
    # Create datasets in a 'data' directory
    data_dir = Path("data")
    create_sample_datasets(data_dir)

    # Print information about created datasets
    for file in data_dir.glob("*.csv"):
        print_dataset_info(file)