# RDF RAG Agent 🤖

A Retrieval-Augmented Generation (RAG) application built with LangChain and Streamlit for querying RGPV RDF Guidelines documents using natural language.


## 🌟 Features

- **Intelligent Document Search**: Uses vector embeddings to find relevant information from RDF/RIKSDF Guidelines
- **Conversational AI**: Maintains chat history for context-aware responses
- **Modern UI**: Clean, responsive Streamlit interface with RGPV branding
- **Efficient Vector Storage**: Powered by Pinecone for fast similarity search
- **Advanced LLM**: Uses Groq's Llama 3.3 70B model for high-quality responses
- **Local Embeddings**: HuggingFace embeddings for privacy and cost efficiency

## 📋 Prerequisites

- Python 3.8 or higher
- Pinecone account and API key
- Google API key (for additional features)
- Groq API key

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/RDF-RAG-Agent.git
cd RDF-RAG-Agent
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

Create a `.env` file in the root directory:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```env
PINECONE_API_KEY=your_pinecone_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Create Vector Database

Run the following command to process the PDF and create the vector database:

```bash
python create_vector_db.py
```

This will:
- Load the RDF Guidelines PDF
- Split it into chunks (chunk_size=1000, overlap=200)
- Generate embeddings using HuggingFace's all-MiniLM-L6-v2 model
- Store vectors in Pinecone

### 5. Run the Application

**Windows:**
```bash
run.bat
```

**Linux/Mac:**
```bash
export TOKENIZERS_PARALLELISM=false
streamlit run streamlit_app.py
```

The application will open in your default browser at `http://localhost:8501`

## 📁 Project Structure

```
RDF-RAG-Agent/
├── app.py                      # Main Streamlit application
├── streamlit_app.py            # Alternative Streamlit entry point
├── create_vector_db.py         # Script to create Pinecone vector database
├── query_rag.py                # RAG chain implementation
├── run.bat                     # Windows startup script
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (not in repo)
├── .env.example                # Environment variables template
├── .gitignore                  # Git ignore rules
├── image/                      # Logo and images
│   └── RGPV Logo.png
├── libs/                       # Local Python libraries (not in repo)
├── RDF_RIKSDF_Guidelines_2022_23.pdf  # Source document
└── README.md                   # This file
```

## 🔧 Configuration

### Vector Database Settings

In `create_vector_db.py`:
- **Chunk Size**: 1000 characters
- **Chunk Overlap**: 200 characters
- **Embedding Model**: all-MiniLM-L6-v2 (384 dimensions)
- **Vector DB**: Pinecone (AWS us-east-1, cosine similarity)
- **Max Concurrency**: 5 threads

### LLM Settings

In `query_rag.py`:
- **Model**: Llama 3.3 70B Versatile (via Groq)
- **Temperature**: 0 (deterministic)
- **Retrieval**: Top 10 similar chunks
- **Max Retries**: 2

## 💡 Usage

1. **Ask Questions**: Type your question in the chat input
2. **View Responses**: The AI will search the document and provide concise answers
3. **Follow-up Questions**: The system maintains conversation history for context

Example queries:
- "What is PSDC?"
- "How is the committee constituted?"
- "What are the RDF guidelines for 2022-23?"

## 🛠️ Development

### Testing the RAG Chain

```bash
python query_rag.py
```

This runs a test query to verify the RAG chain is working correctly.

### Debugging

Enable verbose logging by setting environment variable:
```bash
set LANGCHAIN_VERBOSE=true
```

## 📦 Dependencies

Key libraries:
- **Streamlit**: Web interface
- **LangChain**: RAG framework
- **Pinecone**: Vector database
- **HuggingFace**: Embeddings
- **Groq**: LLM API
- **PyPDF**: PDF processing

See `requirements.txt` for complete list.

## 🔐 Security Notes

- Never commit your `.env` file to version control
- Keep your API keys secure
- The `.gitignore` file is configured to exclude sensitive files
- Consider using environment-specific `.env` files for production

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- Your Name - Initial work

## 🙏 Acknowledgments

- RGPV for the RDF Guidelines documentation
- LangChain community for the excellent framework
- Pinecone for vector database services
- Groq for LLM API access

## 📞 Support

For issues and questions:
- Open an issue on GitHub
- Contact: prashantsteele@gmail.com

## 🔄 Version History

- **v1.0.0** (2025-12-17)
  - Initial release
  - RAG implementation with Pinecone
  - Streamlit web interface
  - Chat history support
  - HuggingFace embeddings integration

---

Made with ❤️ for RGPV
