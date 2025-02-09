from typing import Dict, List, Any
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
import logging

from .vector_store import DatasetVectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetQueryEngine:
    def __init__(self, model_name: str = "gpt-3.5-turbo-16k"):
        """
        Initialize the query engine.

        Args:
            model_name: Name of the LLM model to use
        """
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=0,
        )
        self.vector_store = DatasetVectorStore()
        self.df = None
        self.metadata = None

        # Initialize the base prompt template
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful data analyst assistant. Answer questions about the dataset using the provided context and data.
            Always provide accurate, data-driven responses based on the actual values in the dataset.
            If you need to perform calculations, explain your methodology.
            If you cannot answer a question with the available information, say so clearly.

            Dataset Context:
            {context}

            Additional Information:
            {relevant_docs}
            """),
            ("human", "{question}")
        ])

        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt_template,
            verbose=True
        )

    def load_dataset(self, df: pd.DataFrame, metadata: Dict[str, Any]) -> None:
        """
        Load a dataset and create its vector store.

        Args:
            df: Input DataFrame
            metadata: Dataset metadata
        """
        self.df = df
        self.metadata = metadata
        self.vector_store.create_vector_store(df, metadata)
        logger.info("Dataset loaded and vector store created")

    def _prepare_context(self, query: str) -> Dict[str, str]:
        """
        Prepare context for the query using vector store retrieval.

        Args:
            query: User question

        Returns:
            Dict containing context and relevant documents
        """
        # Get relevant documents from vector store
        relevant_docs = self.vector_store.similarity_search(query)

        # Basic dataset context
        context = f"""This dataset contains {self.metadata['num_rows']} rows and {self.metadata['num_columns']} columns.
        The columns are: {', '.join(self.df.columns)}."""

        # Format relevant documents
        docs_text = "\n\n".join([
            f"Document {i+1}:\n{doc.page_content}"
            for i, doc in enumerate(relevant_docs)
        ])

        return {
            "context": context,
            "relevant_docs": docs_text
        }

    def _execute_query(self, query: str, context: Dict[str, str]) -> str:
        """
        Execute the query using the LLM chain.

        Args:
            query: User question
            context: Prepared context

        Returns:
            Generated response
        """
        try:
            response = self.chain.run(
                question=query,
                **context
            )
            return response
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            raise

    def query(self, question: str) -> str:
        """
        Process a user question and generate a response.

        Args:
            question: User question

        Returns:
            Generated response
        """
        if self.df is None or self.vector_store.vector_store is None:
            raise ValueError("Dataset not loaded. Call load_dataset first.")

        try:
            # Prepare context for the query
            context = self._prepare_context(question)

            # Execute query and get response
            response = self._execute_query(question, context)

            return response

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return f"Error processing your question: {str(e)}"

    def save_state(self, vector_store_path: str) -> None:
        """Save the query engine state"""
        self.vector_store.save_vector_store(vector_store_path)

    def load_state(self, vector_store_path: str) -> None:
        """Load the query engine state"""
        self.vector_store.load_vector_store(vector_store_path)