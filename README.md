<img width="860" height="378" alt="Screenshot 2026-08-22 025556" src="https://github.com/user-attachments/assets/57e2837f-865e-4989-9388-dd33059c4bad" /># Atlas
<img width="1920" height="1080" alt="atlas" src="https://github.com/user-attachments/assets/fa3a3286-7827-4322-9f7b-c4df9e55b677" />

Atlas RAG is a Retrieval-Augmented Generation (RAG) application built with Python, LangChain, and Streamlit. It lets users upload PDF documents, split them into searchable chunks, store them in a vector database, and ask questions about the content using a language model.

This project is designed for document Q&A workflows, research assistants, and knowledge retrieval from PDF-based sources.

## Features

- PDF upload and ingestion
- Text chunking and embedding generation
- Vector store creation using ChromaDB or FAISS
- Semantic retrieval for relevant document chunks
- Question answering on extracted knowledge
- Streamlit-based web interface
- Groq model integration for fast LLM responses

## Tech Stack

- Python
- Streamlit
- LangChain
- LangChain Community
- ChromaDB
- FAISS
- PyMuPDF / PyPDF
- Sentence Transformers
- Groq
- Python Dotenv

## Project Structure

```bash
Atlas Rag/
├── data/
│   └── vector_store/
├── notebook/
│   ├── document.ipynb
│   └── pdf_loader.ipynb
├── src/
│   └── atlas_rag/
├── .env
├── .gitignore
├── main.py
├── pyproject.toml
├── requirements.txt
├── README.md
└── uv.lock
```

## Prerequisites

Before running the project, make sure you have:

- Python 3.10 or later
- pip or uv installed
- A Groq API key
- A working virtual environment

## Environment Variables

Create a `.env` file in the project root and add your keys:

```env
GROQ_API_KEY=your_groq_api_key_here
```

If your app uses other environment variables, add them here as well.

## Installation

### Option 1: Using pip

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Option 2: Using uv

```bash
uv sync
```

## Run the App Locally

Start the Streamlit app with:

```bash
streamlit run main.py
```

Then open the local URL shown in the terminal, usually:

```bash
http://localhost:8501
```

## How It Works

1. Upload a PDF file or add documents to the project.
2. The app extracts text from the PDF.
3. Text is split into smaller chunks.
4. Chunks are embedded and stored in a vector database.
5. The user asks a question in the Streamlit interface.
6. Relevant chunks are retrieved and passed to the LLM.
7. The model generates a context-based answer.

## Pipeline Structure
<img width="847" height="707" alt="Screenshot 2026-08-22 163115" src="https://github.com/user-attachments/assets/e9fa9b28-2c71-4826-8257-f838c325ce07" />

<img width="860" height="378" alt="Screenshot 2026-08-22 025556" src="https://github.com/user-attachments/assets/f06f09c2-cdf4-4322-8371-1a35c11fcab5" />

## Streamlit Deployment

This project is intended to run as a Streamlit application and can be deployed using Streamlit Community Cloud or another hosting provider.

### Deploy to Streamlit Cloud

1. Push the project to GitHub.
2. Go to Streamlit Cloud.
3. Create a new app.
4. Select the repository and branch.
5. Set the app main file to `main.py`.
6. Add environment variables such as `GROQ_API_KEY`.
7. Deploy the app.

## Notes

- The vector database files are stored under `data/vector_store/`.
- If you want to reset the index, delete the stored vector database files and rerun the app.
- For production use, consider adding better error handling, file validation, and caching.

## Example Use Cases

- Academic paper search
- Internal document Q&A
- Research knowledge assistant
- PDF-based support bot

## License

This project is for educational and personal use. Add your chosen license if you plan to share it publicly.
