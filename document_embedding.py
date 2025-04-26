from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from pypdf import PdfReader
import tiktoken
import uuid

PDF_PATH = ""
OPENAI_API_KEY = ""
COLLECTION_NAME = "rag_pdf_docs_2"
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-4o-mini"  
MAX_EMBEDDING_TOKENS = 8191
CHUNK_SIZE = 500  
CONTEXT_TOKEN_LIMIT = 6000 

client = OpenAI(api_key=OPENAI_API_KEY)
qdrant = QdrantClient(host="localhost", port=6333)

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def chunk_text(text, max_tokens=CHUNK_SIZE):
    encoding = tiktoken.encoding_for_model(EMBEDDING_MODEL)
    words = text.split()
    chunks, chunk, token_count = [], [], 0

    for word in words:
        token_count += len(encoding.encode(word))
        chunk.append(word)
        if token_count >= max_tokens:
            chunks.append(" ".join(chunk))
            chunk, token_count = [], 0

    if chunk:
        chunks.append(" ".join(chunk))
    return chunks

def embed_text(texts, batch_size=20):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch
        )
        batch_embeddings = [d.embedding for d in response.data]
        embeddings.extend(batch_embeddings)
    return embeddings

pdf_text = extract_text_from_pdf(PDF_PATH)
documents = chunk_text(pdf_text)
embeddings = embed_text(documents)

qdrant.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
)

points = [
    PointStruct(id=str(uuid.uuid4()), vector=vec, payload={"text": doc})
    for doc, vec in zip(documents, embeddings)
]

qdrant.upsert(collection_name=COLLECTION_NAME, points=points)



