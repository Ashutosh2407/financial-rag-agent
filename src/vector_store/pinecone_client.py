import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

def get_index(name: str | None, dimension: int = 1536):
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index_name = name or os.environ.get("PINECONE_INDEX_NAME", "financial-rag")
    if index_name not in pc.list_indexes():
        pc.create_index(
            name = index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return pc.Index(index_name)


