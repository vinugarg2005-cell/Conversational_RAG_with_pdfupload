Conversational RAG Chatbot with PDF Upload & Chat History
Overview
A conversational AI chatbot that lets you upload PDF documents and chat with their content. Built with LangChain, Groq, HuggingFace, Chroma, and Streamlit. Supports multi-turn conversations with full chat history awareness.

Features

Upload any PDF and chat with its content
History-aware retrieval — understands follow-up questions
Super fast inference via Groq (Llama 3.3 70B)
Semantic search using HuggingFace embeddings
Session-based chat history — multiple users supported
Clean Streamlit UI


Tech Stack
ToolPurposeLangChainRAG pipeline & chain orchestrationGroqLLM inference (Llama 3.3 70B)HuggingFaceText embeddings (all-MiniLM-L6-v2)ChromaVector databaseStreamlitWeb UIPyPDFPDF loading

Setup & Installation
1. Clone the repo
bashgit clone https://github.com/yourusername/rag-chatbot-history.git
cd rag-chatbot-history
2. Create virtual environment (Python 3.11 recommended)
bashpython3.11 -m venv venv311
source venv311/bin/activate
3. Install dependencies
bashpip install -r requirements.txt
4. Setup environment variables
bashcp .env.example .env
Fill in your API keys in .env:
HF_TOKEN=your_huggingface_token
GROQ_API_KEY=your_groq_api_key
Get your keys here:

HuggingFace token → https://huggingface.co/settings/tokens
Groq API key → https://console.groq.com

5. Run the app
bashstreamlit run app.py

 How It Works
User uploads PDF
      ↓
PDF → chunks → HuggingFace embeddings → Chroma vector DB
      ↓
User asks question
      ↓
Chat history + question → LLM → standalone question
      ↓
Standalone question → Chroma → relevant chunks
      ↓
Chunks + history + question → LLM → final answer

 Project Structure
├── app.py              # Main Streamlit app
├── requirements.txt    # Dependencies
├── .env.example        # Sample environment variables
├── .gitignore          # Git ignore rules
└── README.md           # This file

 Usage

Run the app
Enter your Groq API key in the sidebar
Enter a Session ID (default: default_session)
Upload a PDF file
Ask questions about the PDF content
Chat naturally — the bot remembers previous messages!


 Requirements

Python 3.11
Groq API key (free at console.groq.com)
HuggingFace token (free at huggingface.co)


