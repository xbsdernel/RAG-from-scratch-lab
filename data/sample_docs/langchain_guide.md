# LangChain Framework Guide

## Overview

LangChain is an open-source framework for building applications powered by large language models (LLMs). It provides abstractions for common LLM application patterns including chains, agents, memory, and retrieval.

LangChain supports multiple LLM providers (OpenAI, Anthropic, Google Gemini, Cohere, Ollama, etc.) and vector stores (ChromaDB, FAISS, Pinecone, Weaviate) through a unified interface.

## Key Components

### Documents and Document Loaders

A `Document` in LangChain has two fields:
- `page_content`: The text content.
- `metadata`: A dictionary with arbitrary key-value pairs (source, page, etc.).

Document loaders parse files (PDF, DOCX, HTML, CSV, etc.) into lists of `Document` objects.

### Text Splitters

Text splitters divide documents into smaller chunks. The most commonly used is `RecursiveCharacterTextSplitter`, which tries to split on natural boundaries:
1. Double newlines (paragraph breaks)
2. Single newlines
3. Periods
4. Spaces
5. Individual characters (last resort)

### Embeddings

LangChain provides `Embeddings` base class with two methods:
- `embed_documents(texts)`: Embed a list of document passages.
- `embed_query(text)`: Embed a single search query.

### Vector Stores

Vector stores manage the storage and retrieval of embeddings. The `Chroma` class from `langchain-chroma` provides a persistent local vector database backed by ChromaDB. Key methods:
- `add_documents(docs)`: Add documents with automatic embedding.
- `similarity_search(query, k)`: Retrieve top-k similar documents.
- `similarity_search_with_score(query, k)`: Retrieve with distances.
- `similarity_search_by_vector(embedding, k)`: Search by pre-computed vector.

### LangChain Expression Language (LCEL)

LCEL provides a pipe-based syntax for composing chains:

```python
chain = prompt | llm | output_parser
result = chain.invoke({"question": "..."})
```

LCEL supports streaming, batching, and async execution.

### Prompt Templates

`ChatPromptTemplate` creates structured prompts with system and human messages:

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{question}")
])
```

## Integration Packages

LangChain uses separate packages for integrations:
- `langchain-google-genai`: Google Gemini models via `ChatGoogleGenerativeAI`.
- `langchain-chroma`: ChromaDB vector store.
- `langchain-huggingface`: HuggingFace embeddings via `HuggingFaceEmbeddings`.
- `langchain-community`: Community integrations including Ollama.

## Google Gemini Integration

```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key="YOUR_API_KEY",
    temperature=0.0,
)
```

The Gemini free tier (available via Google AI Studio) supports `gemini-2.0-flash` and related models with rate limits. Check current limits at aistudio.google.com.

## ChromaDB Integration

ChromaDB stores vectors persistently on disk. The collection persists across application restarts:

```python
from langchain_chroma import Chroma

vectorstore = Chroma(
    collection_name="my_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)
```

Do not delete the persist directory between runs unless you want to clear the index.

## Error Handling

Common errors in LangChain applications:
- `429 RESOURCE_EXHAUSTED`: Rate limit exceeded. Implement exponential backoff.
- `AuthenticationError`: Invalid API key.
- `InvalidRequestError`: Malformed prompt or unsupported parameters.
- Import errors: Missing integration packages (install with pip).
