import streamlit as st
from pathlib import Path
import pandas as pd
from typing import Tuple
import time
import logging
from functools import lru_cache

from .data_loader import load_csv, generate_data_summary
from .query_engine import DatasetQueryEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'query_engine' not in st.session_state:
        st.session_state.query_engine = DatasetQueryEngine()
    if 'current_file' not in st.session_state:
        st.session_state.current_file = None

@lru_cache(maxsize=2)  # Cache for both small and large datasets
def load_cached_dataset(file_path: str) -> Tuple[pd.DataFrame, dict]:
    """
    Load and cache dataset to avoid reloading on every rerun.

    Args:
        file_path: Path to CSV file

    Returns:
        Tuple of DataFrame and metadata
    """
    return load_csv(file_path)

def display_dataset_info(df: pd.DataFrame, metadata: dict):
    """Display basic dataset information"""
    with st.expander("Dataset Information", expanded=False):
        st.write(f"Rows: {metadata['num_rows']}")
        st.write(f"Columns: {metadata['num_columns']}")
        st.write("Sample Data:")
        st.dataframe(df.head(5), use_container_width=True)

        if st.checkbox("Show Detailed Statistics"):
            st.write("Dataset Statistics:")
            st.write(generate_data_summary(df))

def handle_file_upload():
    """Handle file upload and dataset loading"""
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

    if uploaded_file:
        try:
            # Save uploaded file temporarily
            temp_path = Path("temp_upload.csv")
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.getvalue())

            # Load and cache dataset
            df, metadata = load_cached_dataset(str(temp_path))

            # Update session state
            if st.session_state.current_file != str(temp_path):
                st.session_state.current_file = str(temp_path)
                st.session_state.query_engine.load_dataset(df, metadata)
                st.session_state.messages = []  # Clear chat history for new dataset

            return df, metadata

        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return None, None
    return None, None

def display_chat_interface():
    """Display chat messages and handle new messages"""
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input
    if prompt := st.chat_input("Ask a question about the dataset"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    start_time = time.time()
                    response = st.session_state.query_engine.query(prompt)
                    end_time = time.time()

                    st.markdown(response)
                    st.caption(f"Response time: {(end_time - start_time):.2f} seconds")

                    # Add assistant message
                    st.session_state.messages.append({"role": "assistant", "content": response})

                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

def main():
    """Main Streamlit application"""
    st.title("CSV Dataset Query Assistant")

    # Initialize session state
    init_session_state()

    # Sidebar
    with st.sidebar:
        st.header("Settings")

        # Dataset loading
        st.subheader("Load Dataset")
        df, metadata = handle_file_upload()

        if df is not None:
            display_dataset_info(df, metadata)

        # Clear chat button
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

    # Main chat interface
    if st.session_state.current_file:
        display_chat_interface()
    else:
        st.info("Please upload a CSV file to start the conversation.")

if __name__ == "__main__":
    main()