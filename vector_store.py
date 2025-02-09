from typing import Dict, List, Any
import pandas as pd
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetVectorStore:
    def __init__(self, embedding_model=None):
        """
        Initialize the vector store for dataset chunks.

        Args:
            embedding_model: Optional custom embedding model (defaults to OpenAI)
        """
        self.embedding_model = embedding_model or OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", ", ", " "]
        )
        self.vector_store = None

    def _create_dataset_chunks(self, df: pd.DataFrame, metadata: Dict[str, Any]) -> List[Document]:
        """
        Create document chunks from the dataset for vectorization.

        Args:
            df: Input DataFrame
            metadata: Dataset metadata

        Returns:
            List of Document objects
        """
        documents = []

        # Create overview chunk
        overview = f"""Dataset Overview:
        - Total rows: {metadata['num_rows']}
        - Total columns: {metadata['num_columns']}
        - Columns: {', '.join(df.columns)}
        """
        documents.append(Document(
            page_content=overview,
            metadata={'chunk_type': 'overview'}
        ))

        # Create column-specific chunks
        for column in df.columns:
            col_info = metadata['schema'][column]
            col_text = f"""Column: {column}
            - Data type: {col_info['dtype']}
            - Unique values: {col_info['num_unique']}
            - Missing values: {col_info['num_missing']}
            """

            # Add type-specific statistics
            if np.issubdtype(df[column].dtype, np.number):
                stats = df[column].describe()
                col_text += f"""
                - Mean: {stats['mean']:.2f}
                - Median: {stats['50%']:.2f}
                - Std Dev: {stats['std']:.2f}
                - Range: {stats['min']:.2f} to {stats['max']:.2f}
                """
            elif df[column].dtype == 'object':
                top_values = df[column].value_counts().head(5)
                col_text += "\nTop 5 values:\n"
                for val, count in top_values.items():
                    col_text += f"- {val}: {count} occurrences\n"

            documents.append(Document(
                page_content=col_text,
                metadata={'chunk_type': 'column_info', 'column': column}
            ))

        # Create chunks for sample data
        sample_data = df.head(5).to_string()
        sample_chunks = self.text_splitter.create_documents(
            [sample_data],
            metadatas=[{'chunk_type': 'sample_data'}]
        )
        documents.extend(sample_chunks)

        return documents

    def create_vector_store(self, df: pd.DataFrame, metadata: Dict[str, Any]) -> None:
        """
        Create and store vector embeddings for the dataset.

        Args:
            df: Input DataFrame
            metadata: Dataset metadata
        """
        try:
            logger.info("Creating document chunks...")
            documents = self._create_dataset_chunks(df, metadata)

            logger.info("Creating vector store...")
            self.vector_store = FAISS.from_documents(
                documents,
                self.embedding_model
            )
            logger.info("Vector store created successfully")

        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Perform similarity search for a query.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of relevant Document objects
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call create_vector_store first.")

        return self.vector_store.similarity_search(query, k=k)

    def save_vector_store(self, path: str) -> None:
        """Save the vector store to disk"""
        if self.vector_store is None:
            raise ValueError("No vector store to save")
        self.vector_store.save_local(path)

    def load_vector_store(self, path: str) -> None:
        """Load the vector store from disk"""
        self.vector_store = FAISS.load_local(path, self.embedding_model)