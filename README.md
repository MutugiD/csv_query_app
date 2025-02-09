# CSV Dataset Query Assistant

An intelligent chatbot that helps you analyze and query CSV datasets using natural language, and more able to answer some statistical questions. Built with Streamlit, LangChain, and OpenAI's GPT models.

Refining the setup for average to advanced statistical queries. 

## Features

- 🔍 Natural language queries for CSV data analysis
- 📊 Interactive data visualization and statistics
- 💾 Efficient data loading with caching
- 🔄 Vector-based similarity search for context-aware responses
- ⚡ Support for both small and large datasets (tested up to 50,000 rows)
- 🎯 Accurate responses with data-driven context

## Installation

1. Clone the repository:
```git clone https://github.com/MutugiD/csv_query_app```

Then cd into the csv_query_app directory:
```cd csv_query_app```


2. Install dependencies:
```pip install -r requirements.txt```
3. Set up environment variables:

Create a `.env` file in the root directory with your OpenAI API key:
```OPENAI_API_KEY=your_openai_api_key```

## Usage
1. Run the Streamlit app:
```python -m csv_query.run```

2. Upload your CSV file and interact with the chatbot.

## Customization

- Modify the `data_loader.py` to customize data loading and preprocessing.
- Update the `query_engine.py` to change the RAG prompt and model.

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.



