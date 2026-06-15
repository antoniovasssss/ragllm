import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.documents import Document

load_dotenv(override=True)

MODEL = "gpt-4.1-nano"
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Connect to Pinecone index + namespace
index_name = "insurellm-main"
# Namespace to separate datasets
NAMESPACE = "insurellm"

vectorstore = PineconeVectorStore(
    index_name=index_name,
    embedding=embeddings,
    namespace=NAMESPACE
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

llm = ChatOpenAI(temperature=0, model_name=MODEL)

SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing Insurellm.
Use the retrieved context to answer questions.
If you don’t know, say so.
Context:
{context}
"""


def resolve_query(query: str, history: list):
    last_user = None
    if history:
        for turn in reversed(history):
            if turn["role"] == "user":
                last_user = turn["content"]
                break
    if last_user:
        return f"{last_user}\n{query}"
    return query


def fetch_context(query: str) -> list[Document]:
    return retriever.invoke(query)


def answer_question(query: str, history=None):
    resolved_query = resolve_query(query, history or [])
    docs = fetch_context(resolved_query)
    context = "\n\n".join(doc.page_content for doc in docs)

    messages = [SystemMessage(content=SYSTEM_PROMPT.format(context=context))]

    if history:
        for turn in history:
            if turn["role"] == "user":
                messages.append(HumanMessage(content=turn["content"]))
            elif turn["role"] == "assistant":
                messages.append(AIMessage(content=turn["content"]))

    messages.append(HumanMessage(content=query))
    response = llm.invoke(messages)
    return response.content, docs