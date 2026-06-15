import os
import glob
from dotenv import load_dotenv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_community.document_loaders import DirectoryLoader, TextLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

load_dotenv(override=True)

KNOWLEDGE_BASE = str(Path(__file__).parent.parent / "knowledge-base")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Reuse one index (already created in your Pinecone project)
index_name = "insurellm-main"

# Create if missing
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Now connect
index = pc.Index(index_name)


# Namespace to separate datasets
NAMESPACE = "insurellm"


def fetch_documents():
    documents = []
    for folder in glob.glob(str(Path(KNOWLEDGE_BASE) / "*")):
        doc_type = os.path.basename(folder)
        loader = DirectoryLoader(
            folder,
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        for doc in loader.load():
            doc.metadata["doc_type"] = doc_type
            documents.append(doc)
    return documents


def create_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    return splitter.split_documents(documents)


def embed_and_prepare(chunks):
    vectors = []
    for i, doc in enumerate(chunks):
        vec = embeddings.embed_query(doc.page_content)
        vectors.append({
            "id": f"doc-{i}",
            "values": vec,
            "metadata": {
                "source": doc.metadata.get("source", ""),
                "text": doc.page_content,
                "doc_type": doc.metadata.get("doc_type", "")
            }
        })
    return vectors


def insert_parallel(vectors, batch_size=100):
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(
                index.upsert,
                vectors=vectors[i:i + batch_size],
                namespace=NAMESPACE,
            )
            for i in range(0, len(vectors), batch_size)
        ]
        for future in as_completed(futures):
            future.result()  # re-raise any errors from the worker threads


if __name__ == "__main__":
    docs = fetch_documents()
    chunks = create_chunks(docs)
    vectors = embed_and_prepare(chunks)
    insert_parallel(vectors)
    print(f"🚀 Ingestion complete: {len(vectors)} vectors inserted into Pinecone (namespace={NAMESPACE})")