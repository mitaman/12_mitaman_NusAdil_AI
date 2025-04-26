from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import JSONResponse
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import tiktoken
import uuid

OPENAI_API_KEY = ""
COLLECTION_NAME = "rag_pdf_docs_2"
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-4o-mini"
CONTEXT_TOKEN_LIMIT = 6000

client = OpenAI(api_key=OPENAI_API_KEY)
qdrant = QdrantClient(host="localhost", port=6333)
def embed_text(texts, batch_size=20):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch
        )
        batch_embeddings = [d.embedding for d in response.data]
        embeddings.extend(batch_embeddings)
    return embeddings
def rag_query(question: str, top_k: int = 3):
    try:
        encoding = tiktoken.encoding_for_model(GPT_MODEL)
        query_vector = embed_text([question])[0]
        search_result = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k
        )
        context_chunks = []
        total_tokens = 0
        for hit in search_result:
            chunk = hit.payload["text"]
            tokens = len(encoding.encode(chunk))
            if total_tokens + tokens > CONTEXT_TOKEN_LIMIT:
                break
            context_chunks.append(chunk)
            total_tokens += tokens
        context_str = "\n".join(context_chunks)
        prompt = f"""Answer the question based on the context below.
Context:
{context_str}
Question:
{question}
Answer:"""
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
app = FastAPI()
@app.post("/ask")
async def ask_question(query: str = Form(...)):
    answer = rag_query(query)
    return JSONResponse(content={
        "question": query,
        "answer": answer
    })