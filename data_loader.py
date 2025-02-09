import pandas as pd
from typing import Dict, Any, Tuple
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_csv(file_path: str | Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load and analyze a CSV file, returning the dataframe and metadata.

    Args:
        file_path: Path to the CSV file

    Returns:
        Tuple containing:
        - DataFrame: The loaded CSV data
        - Dict: Metadata about the dataset
    """
    try:
        # Load CSV with automatic type inference
        df = pd.read_csv(file_path)

        # Generate dataset metadata
        metadata = {
            'num_rows': len(df),
            'num_columns': len(df.columns),
            'column_types': df.dtypes.to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
            'schema': {
                col: {
                    'dtype': str(df[col].dtype),
                    'num_unique': df[col].nunique(),
                    'num_missing': df[col].isna().sum()
                } for col in df.columns
            }
        }

        logger.info(f"Successfully loaded CSV with {metadata['num_rows']} rows and {metadata['num_columns']} columns")
        return df, metadata

    except Exception as e:
        logger.error(f"Error loading CSV file: {str(e)}")
        raise

def generate_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a comprehensive summary of the dataset.

    Args:
        df: Input DataFrame

    Returns:
        Dict containing dataset summary statistics
    """
    summary = {
        'basic_stats': df.describe().to_dict(),
        'correlations': df.select_dtypes(include=['number']).corr().to_dict(),
        'column_summaries': {}
    }

    for column in df.columns:
        col_data = df[column]
        col_summary = {
            'dtype': str(col_data.dtype),
            'num_unique': col_data.nunique(),
            'num_missing': col_data.isna().sum(),
        }

        # Add type-specific statistics
        if pd.api.types.is_numeric_dtype(col_data):
            col_summary.update({
                'mean': col_data.mean(),
                'median': col_data.median(),
                'std': col_data.std(),
                'skew': col_data.skew()
            })
        elif pd.api.types.is_string_dtype(col_data):
            col_summary.update({
                'most_common': col_data.value_counts().head(5).to_dict(),
                'avg_length': col_data.str.len().mean()
            })

        summary['column_summaries'][column] = col_summary

    return summary

def prepare_context(df: pd.DataFrame, metadata: Dict[str, Any]) -> str:
    """
    Prepare a context string about the dataset for the LLM.

    Args:
        df: Input DataFrame
        metadata: Dataset metadata

    Returns:
        String containing dataset context
    """
    context = [
        f"This dataset contains {metadata['num_rows']} rows and {metadata['num_columns']} columns.",
        "\nColumns and their types:",
    ]

    for col, info in metadata['schema'].items():
        context.append(f"- {col} ({info['dtype']}): {info['num_unique']} unique values, {info['num_missing']} missing values")

    return "\n".join(context)