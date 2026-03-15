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
import os

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile")


def extract_video_id(url):
    parsed = urlparse(url)
    
    if "youtu.be" in parsed.netloc:
        return parsed.path.strip("/")
    
    elif "youtube.com" in parsed.netloc:
        query_params = parse_qs(parsed.query)
        return query_params["v"][0]

# Test
url = input("enter the url : ")
video_id = extract_video_id(url)
print("Video ID:", video_id)

# Fetch transcript
ytt = YouTubeTranscriptApi()
transcript_data = ytt.fetch(video_id , languages=['en' , 'hi'])
transcript = " ".join(chunk.text for chunk in transcript_data)
print(transcript[:500])


splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_text(transcript)
print(f"Total chunks: {len(chunks)}")
print(chunks[0])


embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

vector_store = FAISS.from_texts(chunks, embedding=embeddings)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

prompt = ChatPromptTemplate.from_template("""
Answer the question based on the context below.
Respond in the same language as the question. If the question is in Hindi, answer in Hindi. If in English, answer in English.

Context: {context}
Question: {question}
""")

chain = (
    {"context": retriever, "question": RunnablePassthrough()}  | prompt   | llm | StrOutputParser())

while True:
    question = input("enter query (or 'exit' to quit): ")
    if question == "exit":
        break
    response = chain.invoke(question)
    print(response)