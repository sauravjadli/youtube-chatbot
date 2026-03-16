# YouTube Chatbot 🎥

A RAG-based chatbot that lets you ask questions about any YouTube video in any language — Hindi, English, or Hinglish.

## How it works
1. Paste a YouTube video URL
2. The app fetches the video transcript
3. Ask any question about the video
4. Get instant answers in the same language you asked

## Project Structure
- `app.py` — backend logic (transcript fetching, embeddings, RAG chain)
- `app_ui.py` — frontend UI built with Streamlit

## Prerequisites
- Python 3.12+
- A Google API key (for embeddings) from [aistudio.google.com](https://aistudio.google.com)
- A Groq API key (for LLM) from [console.groq.com](https://console.groq.com)

## Installation
```bash
pip install -r requirements.txt
```

## Setup
Create a `.env` file in the root folder:
```
GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key
```

## Run the app
```bash
streamlit run app_ui.py
```

## Tech Stack
- LangChain
- Google Generative AI Embeddings
- FAISS Vector Store
- Groq LLM (llama-3.3-70b)
- Streamlit
- YouTube Transcript API