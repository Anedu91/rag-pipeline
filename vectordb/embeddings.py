"""
Embeddings module for generating vector embeddings from text chunks.

Pure functional approach - all functions are stateless.
Supports multiple embedding providers through a provider pattern.

Available providers:
- sentence-transformers: Local models (free, no API key)
- openai: OpenAI API (requires API key)

Recommended local models:
- all-MiniLM-L6-v2: Fast, 384 dimensions (default)
- all-mpnet-base-v2: Best quality/speed, 768 dimensions
- BAAI/bge-small-en-v1.5: Good accuracy, 384 dimensions
- BAAI/bge-m3: Multilingual, 1024 dimensions
"""

import os
from dataclasses import dataclass
from typing import Callable, Protocol

from dotenv import load_dotenv

from vectordb.chunking import Chunk

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")

# Common local models with their dimensions
LOCAL_MODELS = {
    "all-MiniLM-L6-v2": 384,
    "all-mpnet-base-v2": 768,
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-m3": 1024,
    "intfloat/e5-small-v2": 384,
    "intfloat/e5-base-v2": 768,
}


@dataclass(frozen=True)
class Embedding:
    """Immutable embedding with associated chunk and metadata."""

    vector: tuple[float, ...]
    chunk: Chunk
    model: str
    dimensions: int


@dataclass
class EmbeddingConfig:
    """
    Configuration for embedding generation.

    Attributes:
        model: Model name (local model or OpenAI model)
        dimensions: Embedding dimensions (auto-detected for local models)
        batch_size: Number of texts per batch
        api_key: API key for OpenAI (not needed for local models)
        device: Device for local models ("cpu", "cuda", "mps")
        normalize: Whether to normalize embeddings
    """

    model: str = "all-MiniLM-L6-v2"
    dimensions: int = 384
    batch_size: int = 100
    api_key: str | None = openai_key
    device: str | None = None
    normalize: bool = True


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        ...

    @property
    def model_name(self) -> str:
        """Return the model name."""
        ...

    @property
    def dimensions(self) -> int:
        """Return the embedding dimensions."""
        ...


def create_openai_provider(config: EmbeddingConfig) -> EmbeddingProvider:
    """Create an OpenAI embedding provider."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package required. Install with: uv add openai")

    api_key = config.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenAI API key required. Set OPENAI_API_KEY env var or pass in config."
        )

    client = OpenAI(api_key=api_key)

    class OpenAIProvider:
        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            response = client.embeddings.create(
                model=config.model, input=texts, dimensions=config.dimensions
            )
            return [item.embedding for item in response.data]

        @property
        def model_name(self) -> str:
            return config.model

        @property
        def dimensions(self) -> int:
            return config.dimensions

    return OpenAIProvider()


def create_sentence_transformers_provider(config: EmbeddingConfig) -> EmbeddingProvider:
    """
    Create a Sentence Transformers embedding provider (local, free).

    This is the recommended provider for most use cases.
    No API key required, runs entirely locally.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers package required. "
            "Install with: uv add sentence-transformers"
        )

    model = SentenceTransformer(config.model, device=config.device)

    class SentenceTransformersProvider:
        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            embeddings = model.encode(
                texts, convert_to_numpy=True, normalize_embeddings=config.normalize
            )
            return [emb.tolist() for emb in embeddings]

        @property
        def model_name(self) -> str:
            return config.model

        @property
        def dimensions(self) -> int:
            return model.get_sentence_embedding_dimension()

    return SentenceTransformersProvider()


def get_provider(
    provider_type: str = "sentence-transformers", config: EmbeddingConfig | None = None
) -> EmbeddingProvider:
    """
    Get an embedding provider by type.

    Args:
        provider_type: "sentence-transformers" (default, local) or "openai"
        config: Embedding configuration

    Returns:
        An EmbeddingProvider instance
    """
    config = config or EmbeddingConfig()

    providers = {
        "openai": create_openai_provider,
        "sentence-transformers": create_sentence_transformers_provider,
        "local": create_sentence_transformers_provider,
    }

    if provider_type not in providers:
        raise ValueError(
            f"Unknown provider: {provider_type}. Available: {list(providers.keys())}"
        )

    return providers[provider_type](config)


def batch_items(items: list, batch_size: int) -> list[list]:
    """Split items into batches of specified size."""
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def embed_texts(
    texts: list[str], provider: EmbeddingProvider, batch_size: int = 100
) -> list[list[float]]:
    """
    Generate embeddings for a list of texts.

    Args:
        texts: List of text strings to embed
        provider: Embedding provider to use
        batch_size: Number of texts per batch

    Returns:
        List of embedding vectors
    """
    if not texts:
        return []

    all_embeddings = []
    batches = batch_items(texts, batch_size)

    for batch in batches:
        embeddings = provider.embed_texts(batch)
        all_embeddings.extend(embeddings)

    return all_embeddings


def embed_chunk(chunk: Chunk, provider: EmbeddingProvider) -> Embedding:
    """
    Generate an embedding for a single chunk.

    Args:
        chunk: The chunk to embed
        provider: Embedding provider to use

    Returns:
        Embedding object with vector and metadata
    """
    vectors = provider.embed_texts([chunk.content])
    vector = vectors[0]

    return Embedding(
        vector=tuple(vector),
        chunk=chunk,
        model=provider.model_name,
        dimensions=provider.dimensions,
    )


def embed_chunks(
    chunks: list[Chunk], provider: EmbeddingProvider, batch_size: int = 100
) -> list[Embedding]:
    """
    Generate embeddings for multiple chunks.

    Args:
        chunks: List of chunks to embed
        provider: Embedding provider to use
        batch_size: Number of chunks per batch

    Returns:
        List of Embedding objects
    """
    if not chunks:
        return []

    texts = [chunk.content for chunk in chunks]
    vectors = embed_texts(texts, provider, batch_size)

    embeddings = []
    for chunk, vector in zip(chunks, vectors):
        embedding = Embedding(
            vector=tuple(vector),
            chunk=chunk,
            model=provider.model_name,
            dimensions=provider.dimensions,
        )
        embeddings.append(embedding)

    return embeddings


def embed_markdown(
    chunks: list[Chunk],
    provider_type: str = "sentence-transformers",
    config: EmbeddingConfig | None = None,
) -> list[Embedding]:
    """
    Main entry point for embedding chunks from markdown files.

    Args:
        chunks: List of Chunk objects (from chunking module)
        provider_type: "sentence-transformers" (default, local) or "openai"
        config: Embedding configuration

    Returns:
        List of Embedding objects

    Examples:
        # Using local model (free, no API key)
        embeddings = embed_markdown(chunks)

        # Using specific local model
        config = EmbeddingConfig(model="all-mpnet-base-v2")
        embeddings = embed_markdown(chunks, config=config)

        # Using OpenAI
        config = EmbeddingConfig(
            model="text-embedding-3-small",
            dimensions=1536
        )
        embeddings = embed_markdown(chunks, "openai", config)
    """
    config = config or EmbeddingConfig()
    provider = get_provider(provider_type, config)
    return embed_chunks(chunks, provider, config.batch_size)


def create_embed_function(
    provider_type: str = "sentence-transformers", config: EmbeddingConfig | None = None
) -> Callable[[str], list[float]]:
    """
    Create a simple embedding function for single texts.

    Useful for query embedding in vector search.

    Args:
        provider_type: "sentence-transformers" or "openai"
        config: Embedding configuration

    Returns:
        Function that takes a string and returns embedding vector

    Example:
        embed_fn = create_embed_function()
        query_vector = embed_fn("What is machine learning?")
    """
    config = config or EmbeddingConfig()
    provider = get_provider(provider_type, config)

    def embed_fn(text: str) -> list[float]:
        vectors = provider.embed_texts([text])
        return vectors[0]

    return embed_fn


def filter_embeddings(
    embeddings: list[Embedding], predicate: Callable[[Embedding], bool]
) -> list[Embedding]:
    """Filter embeddings using a predicate function."""
    return [emb for emb in embeddings if predicate(emb)]


def get_embedding_statistics(embeddings: list[Embedding]) -> dict:
    """Calculate statistics about the embeddings."""
    if not embeddings:
        return {
            "total_embeddings": 0,
            "unique_sources": 0,
            "model": None,
            "dimensions": 0,
        }

    sources = set(emb.chunk.source_file for emb in embeddings)

    return {
        "total_embeddings": len(embeddings),
        "unique_sources": len(sources),
        "model": embeddings[0].model,
        "dimensions": embeddings[0].dimensions,
    }


def list_local_models() -> dict[str, int]:
    """
    List recommended local embedding models with their dimensions.

    Returns:
        Dict mapping model name to embedding dimensions
    """
    return LOCAL_MODELS.copy()
