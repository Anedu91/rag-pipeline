"""
Vector Database module for RAG pipeline.

Pure functional approach for chunking, embedding, and storing markdown documents.

Chunking strategies (from chunking_evaluation library):
- FIXED: Fixed token-based chunks
- RECURSIVE: Hierarchical separator-based (default)
- KAMRADT: Semantic with dynamic breakpoints
- CLUSTER_SEMANTIC: Clustering-based semantic
- LLM_SEMANTIC: LLM-based intelligent chunking

Embedding providers:
- sentence-transformers: Local models, free (default)
- openai: OpenAI API

Vector store backends:
- simple: JSON file-based (good for development)
- pgvector: PostgreSQL with pgvector (production-ready)

RAG (Retrieval Augmented Generation):
- openai: OpenAI models (GPT-4, GPT-3.5-turbo)
- anthropic: Claude models
- ollama: Local models (free)
- groq: Fast inference (free tier)

Usage:
    from vectordb import chunk_markdown, embed_markdown, build_vector_store

    # Chunk a single file or list of files
    chunks = chunk_markdown("data/docs/file.md")
    chunks = chunk_markdown(["file1.md", "file2.md"])

    # Generate embeddings (local by default)
    embeddings = embed_markdown(chunks)

    # Store in vector database
    backend = build_vector_store(embeddings)

    # Query with RAG
    from vectordb import create_embed_function, rag_pipeline
    embed_fn = create_embed_function()
    response = rag_pipeline("What is this about?", backend, embed_fn, "openai")
    print(response.answer)
"""

from vectordb.chunking import (
    Chunk,
    ChunkingConfig,
    ChunkingStrategy,
    chunk_document,
    chunk_documents,
    chunk_markdown,
    chunk_text,
    create_sentence_transformer_embedding_function,
    filter_chunks,
    get_chunk_statistics,
    get_chunker,
    get_openai_embedding_function,
    map_chunks,
)
from vectordb.embeddings import (
    LOCAL_MODELS,
    Embedding,
    EmbeddingConfig,
    create_embed_function,
    embed_chunk,
    embed_chunks,
    embed_markdown,
    embed_texts,
    filter_embeddings,
    get_embedding_statistics,
    get_provider,
    list_local_models,
)
from vectordb.rag import (
    RAGConfig,
    RAGResponse,
    batch_rag_query,
    build_prompt,
    format_context,
    format_response_with_sources,
    generate_answer,
    get_llm_provider,
    rag_pipeline,
    rag_query,
    retrieve_context,
)
from vectordb.vector_store import (
    SearchResult,
    VectorStoreConfig,
    build_vector_store,
    delete_embeddings,
    get_backend,
    get_store_statistics,
    query_vector_store,
    search_similar,
    store_embeddings,
)

__all__ = [
    # Chunking
    "Chunk",
    "ChunkingConfig",
    "ChunkingStrategy",
    "chunk_markdown",
    "chunk_document",
    "chunk_documents",
    "chunk_text",
    "filter_chunks",
    "map_chunks",
    "get_chunk_statistics",
    "get_chunker",
    "get_openai_embedding_function",
    "create_sentence_transformer_embedding_function",
    # Embeddings
    "Embedding",
    "EmbeddingConfig",
    "embed_markdown",
    "embed_chunks",
    "embed_chunk",
    "embed_texts",
    "get_provider",
    "create_embed_function",
    "filter_embeddings",
    "get_embedding_statistics",
    "list_local_models",
    "LOCAL_MODELS",
    # Vector Store
    "SearchResult",
    "VectorStoreConfig",
    "build_vector_store",
    "store_embeddings",
    "search_similar",
    "query_vector_store",
    "get_backend",
    "get_store_statistics",
    "delete_embeddings",
    # RAG
    "RAGConfig",
    "RAGResponse",
    "rag_pipeline",
    "rag_query",
    "get_llm_provider",
    "retrieve_context",
    "format_context",
    "build_prompt",
    "generate_answer",
    "format_response_with_sources",
    "batch_rag_query",
]
