from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs

load_dotenv()

def extract_video_id(url):
    parsed = urlparse(url)
    if "youtu.be" in parsed.netloc:
        return parsed.path.strip("/")
    elif "youtube.com" in parsed.netloc:
        query_params = parse_qs(parsed.query)
        return query_params["v"][0]

def build_chain(url):
    # Step 1: video ID nikalo
    video_id = extract_video_id(url)

    # Step 2: transcript fetch karo
    ytt = YouTubeTranscriptApi()
    transcript_data = ytt.fetch(video_id, languages=['en', 'hi'])
    transcript = " ".join(chunk.text for chunk in transcript_data)

    # Step 3: chunks banao
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(transcript)

    # Step 4: vector store banao
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)

    # Step 5: retriever + chain banao
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant that answers questions about a YouTube video.

Answer ONLY based on the context below. If the answer is not in the context, say you don't know.

Respond in the exact same language and style as the user's question:
- If the question is in Hindi (Devanagari script like यह), reply in Hindi using Devanagari script only — never use Roman script for Hindi words.
- If the question is in Hinglish (like "video kiske baare mein hai"), reply in Hinglish the same way.
- If the question is in pure English, reply in pure English only — no Hindi words at all.

Be conversational and friendly — not robotic.

Context: {context}
Question: {question}
""")

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | ChatGroq(model="llama-3.3-70b-versatile")
        | StrOutputParser()
    )

    return chain