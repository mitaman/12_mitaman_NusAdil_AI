from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import JSONResponse
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import tiktoken

OPENAI_API_KEY = ""  
COLLECTION_NAME = "rag_pdf_docs_2"
EMBEDDING_MODEL = "text-embedding-ada-002"
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
LORA_PATH = "./Model_Lora"  
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONTEXT_TOKEN_LIMIT = 6000

client = OpenAI(api_key=OPENAI_API_KEY)
qdrant = QdrantClient(host="localhost", port=6333)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()

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


def generate_with_qwen(prompt: str, max_tokens=512):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.95
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def rag_query(question: str, top_k: int = 3):
    encoding = tiktoken.get_encoding("cl100k_base")
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

    prompt = f"""You are an intelligent assistant. Answer the question based on the following context.

Context:
{context_str}

Question:
{question}

Answer:"""

    return generate_with_qwen(prompt)


app = FastAPI()

@app.post("/ask")
async def ask_question(query: str = Form(...)):
    try:
        answer = rag_query(query)
        return JSONResponse(content={"question": query, "answer": answer})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
