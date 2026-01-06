"""
Chunking module for splitting markdown documents into smaller pieces.

Pure functional approach - all functions are stateless and side-effect free.
Accepts a single .md file path or a list of .md file paths.

Uses the chunking_evaluation library:
https://github.com/brandonstarxel/chunking_evaluation

Available strategies:
- fixed: Fixed token-based chunks
- recursive: Hierarchical separator-based splitting
- kamradt: Semantic chunking with dynamic breakpoints
- cluster_semantic: Clustering-based semantic chunking
- llm_semantic: LLM-based intelligent chunking
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from chunking_evaluation import BaseChunker
from chunking_evaluation.chunking import (
    ClusterSemanticChunker,
    FixedTokenChunker,
    KamradtModifiedChunker,
    LLMSemanticChunker,
    RecursiveTokenChunker,
)


class ChunkingStrategy(Enum):
    """Available chunking strategies from chunking_evaluation."""

    FIXED = "fixed"
    RECURSIVE = "recursive"
    KAMRADT = "kamradt"
    CLUSTER_SEMANTIC = "cluster_semantic"
    LLM_SEMANTIC = "llm_semantic"


@dataclass(frozen=True)
class Chunk:
    """Immutable chunk of text with metadata."""

    content: str
    source_file: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: dict


@dataclass
class ChunkingConfig:
    """
    Configuration for chunking behavior.

    Attributes:
        strategy: Chunking strategy to use
        chunk_size: Target chunk size in tokens (for fixed/recursive)
        chunk_overlap: Overlap between chunks in tokens (for fixed/recursive)
        avg_chunk_size: Average chunk size for semantic methods
        min_chunk_size: Minimum chunk size for semantic methods
        separators: List of separators for recursive chunking
        embedding_function: Embedding function for semantic chunking
        llm_api_key: API key for LLM-based chunking
        llm_model: Model name for LLM-based chunking
        llm_organisation: Organisation for LLM (openai, anthropic, etc.)
        strip_whitespace: Whether to strip whitespace from chunks
    """

    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE
    chunk_size: int = 512
    chunk_overlap: int = 50
    avg_chunk_size: int = 400
    min_chunk_size: int = 50
    separators: list[str] = field(default_factory=lambda: ["\n\n", "\n", ". ", " "])
    embedding_function: Any = None
    llm_api_key: str | None = None
    llm_model: str | None = None
    llm_organisation: str = "openai"
    strip_whitespace: bool = True


def read_markdown_file(file_path: str | Path) -> str:
    """Read content from a markdown file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if not path.suffix.lower() == ".md":
        raise ValueError(f"Expected .md file, got: {path.suffix}")
    return path.read_text(encoding="utf-8")


def extract_frontmatter(content: str) -> tuple[dict, str]:
    """
    Extract YAML frontmatter from markdown content.

    Returns a tuple of (metadata_dict, remaining_content).
    """
    frontmatter_pattern = r"^---\s*\n(.*?)\n---\s*\n"
    match = re.match(frontmatter_pattern, content, re.DOTALL)

    if not match:
        return {}, content

    frontmatter_text = match.group(1)
    remaining_content = content[match.end() :]

    metadata = {}
    for line in frontmatter_text.split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip()

    return metadata, remaining_content


# =============================================================================
# Chunker Factory Functions
# =============================================================================


def create_fixed_chunker(config: ChunkingConfig) -> BaseChunker:
    """Create a FixedTokenChunker instance."""
    return FixedTokenChunker(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )


def create_recursive_chunker(config: ChunkingConfig) -> BaseChunker:
    """Create a RecursiveTokenChunker instance."""
    return RecursiveTokenChunker(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=config.separators,
    )


def create_kamradt_chunker(config: ChunkingConfig) -> BaseChunker:
    """
    Create a KamradtModifiedChunker instance.

    Requires embedding_function for semantic similarity.
    """
    return KamradtModifiedChunker(
        avg_chunk_size=config.avg_chunk_size,
        min_chunk_size=config.min_chunk_size,
        embedding_function=config.embedding_function,
    )


def create_cluster_semantic_chunker(config: ChunkingConfig) -> BaseChunker:
    """
    Create a ClusterSemanticChunker instance.

    Requires embedding_function for semantic clustering.
    """
    return ClusterSemanticChunker(
        embedding_function=config.embedding_function,
        max_chunk_size=config.chunk_size,
        min_chunk_size=config.min_chunk_size,
    )


def create_llm_semantic_chunker(config: ChunkingConfig) -> BaseChunker:
    """
    Create an LLMSemanticChunker instance.

    Requires LLM API key and model configuration.
    """
    return LLMSemanticChunker(
        organisation=config.llm_organisation,
        api_key=config.llm_api_key,
        model_name=config.llm_model,
    )


def get_chunker(config: ChunkingConfig) -> BaseChunker:
    """
    Get the appropriate chunker based on configuration.

    Args:
        config: Chunking configuration

    Returns:
        A BaseChunker instance

    Raises:
        ValueError: If semantic chunking is requested without embedding_function
    """
    strategy = config.strategy

    if strategy == ChunkingStrategy.FIXED:
        return create_fixed_chunker(config)

    elif strategy == ChunkingStrategy.RECURSIVE:
        return create_recursive_chunker(config)

    elif strategy == ChunkingStrategy.KAMRADT:
        if config.embedding_function is None:
            raise ValueError(
                "Kamradt chunking requires embedding_function. "
                "Use get_openai_embedding_function() or provide your own."
            )
        return create_kamradt_chunker(config)

    elif strategy == ChunkingStrategy.CLUSTER_SEMANTIC:
        if config.embedding_function is None:
            raise ValueError(
                "Cluster semantic chunking requires embedding_function. "
                "Use get_openai_embedding_function() or provide your own."
            )
        return create_cluster_semantic_chunker(config)

    elif strategy == ChunkingStrategy.LLM_SEMANTIC:
        if config.llm_api_key is None:
            raise ValueError("LLM semantic chunking requires llm_api_key.")
        return create_llm_semantic_chunker(config)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# =============================================================================
# Text Position Tracking
# =============================================================================


def find_chunk_positions(
    original_text: str, chunks: list[str]
) -> list[tuple[str, int, int]]:
    """
    Find the start and end positions of chunks in the original text.

    Returns list of tuples: (chunk_content, start_char, end_char)
    """
    positions = []
    search_start = 0

    for chunk in chunks:
        chunk_stripped = chunk.strip()
        start = original_text.find(chunk_stripped, search_start)

        if start == -1:
            first_words = " ".join(chunk_stripped.split()[:5])
            start = original_text.find(first_words, search_start)
            if start == -1:
                start = search_start

        end = start + len(chunk_stripped)
        positions.append((chunk, start, end))
        search_start = start + 1

    return positions


# =============================================================================
# Core Chunking Functions
# =============================================================================


def chunk_text(
    text: str, config: ChunkingConfig | None = None, source_name: str = "inline"
) -> list[Chunk]:
    """
    Chunk raw text directly (not from a file).

    Args:
        text: Text content to chunk
        config: Chunking configuration
        source_name: Name to use for source_file metadata

    Returns:
        List of Chunk objects
    """
    config = config or ChunkingConfig()
    frontmatter, body = extract_frontmatter(text)

    chunker = get_chunker(config)
    raw_chunks = chunker.split_text(body)

    chunks_with_positions = find_chunk_positions(body, raw_chunks)

    chunks = []
    for idx, (chunk_content, start, end) in enumerate(chunks_with_positions):
        content = chunk_content.strip() if config.strip_whitespace else chunk_content
        if not content:
            continue

        chunk = Chunk(
            content=content,
            source_file=source_name,
            chunk_index=idx,
            start_char=start,
            end_char=end,
            metadata={
                "frontmatter": frontmatter,
                "chunk_size": len(content),
                "strategy": config.strategy.value,
            },
        )
        chunks.append(chunk)

    return chunks


def chunk_document(
    file_path: str | Path, config: ChunkingConfig | None = None
) -> list[Chunk]:
    """
    Chunk a single markdown document into smaller pieces.

    Args:
        file_path: Path to the markdown file
        config: Chunking configuration (uses defaults if None)

    Returns:
        List of Chunk objects with content and metadata
    """
    config = config or ChunkingConfig()
    path = Path(file_path)

    content = read_markdown_file(path)
    frontmatter, body = extract_frontmatter(content)

    chunker = get_chunker(config)
    raw_chunks = chunker.split_text(body)

    chunks_with_positions = find_chunk_positions(body, raw_chunks)

    chunks = []
    for idx, (chunk_content, start, end) in enumerate(chunks_with_positions):
        content_final = (
            chunk_content.strip() if config.strip_whitespace else chunk_content
        )
        if not content_final:
            continue

        chunk = Chunk(
            content=content_final,
            source_file=str(path.absolute()),
            chunk_index=idx,
            start_char=start,
            end_char=end,
            metadata={
                "frontmatter": frontmatter,
                "file_name": path.name,
                "chunk_size": len(content_final),
                "strategy": config.strategy.value,
            },
        )
        chunks.append(chunk)

    return chunks


def chunk_documents(
    file_paths: list[str | Path], config: ChunkingConfig | None = None
) -> list[Chunk]:
    """
    Chunk multiple markdown documents.

    Args:
        file_paths: List of paths to markdown files
        config: Chunking configuration (uses defaults if None)

    Returns:
        List of all Chunk objects from all documents
    """
    all_chunks = []
    for path in file_paths:
        try:
            chunks = chunk_document(path, config)
            all_chunks.extend(chunks)
        except (FileNotFoundError, ValueError) as e:
            print(f"Warning: Skipping {path}: {e}")

    return all_chunks


def chunk_markdown(
    source: str | Path | list[str | Path], config: ChunkingConfig | None = None
) -> list[Chunk]:
    """
    Main entry point for chunking markdown files.

    Accepts either a single file path or a list of file paths.

    Args:
        source: Single path or list of paths to markdown files
        config: Chunking configuration (uses defaults if None)

    Returns:
        List of Chunk objects

    Examples:
        # Default recursive chunking
        chunks = chunk_markdown("doc.md")

        # Fixed token chunking
        config = ChunkingConfig(strategy=ChunkingStrategy.FIXED, chunk_size=256)
        chunks = chunk_markdown("doc.md", config)

        # Semantic chunking with embeddings
        from chunking_evaluation import get_openai_embedding_function
        embed_fn = get_openai_embedding_function(api_key="...")
        config = ChunkingConfig(
            strategy=ChunkingStrategy.CLUSTER_SEMANTIC,
            embedding_function=embed_fn
        )
        chunks = chunk_markdown("doc.md", config)
    """
    if isinstance(source, (str, Path)):
        return chunk_document(source, config)
    else:
        return chunk_documents(list(source), config)


# =============================================================================
# Utility Functions
# =============================================================================


def filter_chunks(
    chunks: list[Chunk], predicate: Callable[[Chunk], bool]
) -> list[Chunk]:
    """Filter chunks using a predicate function."""
    return [chunk for chunk in chunks if predicate(chunk)]


def map_chunks(chunks: list[Chunk], transform: Callable[[Chunk], Chunk]) -> list[Chunk]:
    """Apply a transformation to each chunk."""
    return [transform(chunk) for chunk in chunks]


def get_chunk_statistics(chunks: list[Chunk]) -> dict:
    """Calculate statistics about the chunks."""
    if not chunks:
        return {
            "total_chunks": 0,
            "total_characters": 0,
            "avg_chunk_size": 0,
            "min_chunk_size": 0,
            "max_chunk_size": 0,
            "unique_sources": 0,
        }

    sizes = [len(chunk.content) for chunk in chunks]
    sources = set(chunk.source_file for chunk in chunks)

    return {
        "total_chunks": len(chunks),
        "total_characters": sum(sizes),
        "avg_chunk_size": sum(sizes) / len(sizes),
        "min_chunk_size": min(sizes),
        "max_chunk_size": max(sizes),
        "unique_sources": len(sources),
    }


# =============================================================================
# Embedding Function Helpers
# =============================================================================


def get_openai_embedding_function(api_key: str | None = None):
    """
    Get OpenAI embedding function for semantic chunking.

    Args:
        api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)

    Returns:
        Embedding function compatible with chunking_evaluation library
    """
    import os

    from chunking_evaluation import get_openai_embedding_function as _get_openai_ef

    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OpenAI API key required")

    return _get_openai_ef(key)


def create_sentence_transformer_embedding_function(
    model_name: str = "all-MiniLM-L6-v2",
):
    """
    Create a local sentence-transformer embedding function for semantic chunking.

    Args:
        model_name: Name of the sentence-transformers model

    Returns:
        Embedding function compatible with chunking_evaluation library
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers required. Install with: uv add sentence-transformers"
        )

    model = SentenceTransformer(model_name)

    # Note: chunking_evaluation library uses chromadb's EmbeddingFunction interface
    # This function is only used with that library for semantic chunking
    try:
        import chromadb

        class SentenceTransformerEmbeddingFunction(chromadb.EmbeddingFunction):
            def __call__(self, input: list[str]) -> list[list[float]]:
                embeddings = model.encode(input, convert_to_numpy=True)
                return [emb.tolist() for emb in embeddings]

        return SentenceTransformerEmbeddingFunction()
    except ImportError:
        raise ImportError(
            "chromadb required for semantic chunking with chunking_evaluation library. "
            "Install with: uv add chromadb"
        )
