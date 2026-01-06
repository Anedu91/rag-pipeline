"""
Vector Database Pipeline Entry Point.

Build a vector database from markdown files in the /data folder.

Usage:
    # Process a single file (uses local embeddings by default)
    uv run python vectordb_main.py data/docs/file.md

    # Process multiple files
    uv run python vectordb_main.py data/docs/file1.md data/docs/file2.md

    # Process all .md files in a directory
    uv run python vectordb_main.py data/docs/

    # With semantic chunking
    uv run python vectordb_main.py data/docs/ --strategy cluster_semantic

    # Using PostgreSQL with pgvector
    uv run python vectordb_main.py data/docs/ --backend pgvector --pg-connection "postgresql://user:pass@localhost/db"
"""

import argparse
from pathlib import Path

from vectordb import (
    ChunkingConfig,
    ChunkingStrategy,
    EmbeddingConfig,
    VectorStoreConfig,
    build_vector_store,
    chunk_markdown,
    embed_markdown,
    get_chunk_statistics,
    get_embedding_statistics,
    get_store_statistics,
    list_local_models,
)


def find_markdown_files(path: Path) -> list[Path]:
    """Find all markdown files in a path (file or directory)."""
    if path.is_file():
        if path.suffix.lower() == ".md":
            return [path]
        return []
    elif path.is_dir():
        return list(path.rglob("*.md"))
    return []


def collect_files(sources: list[str]) -> list[Path]:
    """Collect all markdown files from the given sources."""
    files = []
    for source in sources:
        path = Path(source)
        files.extend(find_markdown_files(path))
    return files


def print_summary(
    files: list[Path],
    chunk_stats: dict,
    embedding_stats: dict,
    store_stats: dict,
    strategy: str,
    backend: str,
) -> None:
    """Print a summary of the pipeline execution."""
    print("\n" + "=" * 50)
    print("Vector Database Pipeline Summary")
    print("=" * 50)

    print(f"\nFiles processed: {len(files)}")
    for f in files[:5]:
        print(f"  - {f.name}")
    if len(files) > 5:
        print(f"  ... and {len(files) - 5} more")

    print(f"\nChunking ({strategy}):")
    print(f"  Total chunks: {chunk_stats['total_chunks']}")
    print(f"  Total characters: {chunk_stats['total_characters']:,}")
    print(f"  Average chunk size: {chunk_stats['avg_chunk_size']:.0f}")
    print(
        f"  Min/Max chunk size: {chunk_stats['min_chunk_size']}/{chunk_stats['max_chunk_size']}"
    )

    print(f"\nEmbeddings:")
    print(f"  Total embeddings: {embedding_stats['total_embeddings']}")
    print(f"  Model: {embedding_stats['model']}")
    print(f"  Dimensions: {embedding_stats['dimensions']}")

    print(f"\nVector Store ({backend}):")
    print(f"  Total stored: {store_stats['total_embeddings']}")

    print("\n" + "=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Build a vector database from markdown files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single file with local embeddings (free, no API key)
    uv run python vectordb_main.py data/docs/intro.md

    # All files in directory with semantic chunking
    uv run python vectordb_main.py data/docs/ --strategy kamradt

    # Using PostgreSQL pgvector backend
    uv run python vectordb_main.py data/docs/ \\
        --backend pgvector \\
        --pg-connection "postgresql://user:pass@localhost:5432/mydb"

    # Using OpenAI embeddings
    uv run python vectordb_main.py data/docs/ \\
        --provider openai \\
        --model text-embedding-3-small

Chunking Strategies:
    fixed             Fixed token-based chunks
    recursive         Hierarchical separator-based (default)
    kamradt           Semantic with dynamic breakpoints
    cluster_semantic  Clustering-based semantic
    llm_semantic      LLM-based intelligent chunking

Embedding Models (local, free):
    all-MiniLM-L6-v2      Fast, 384 dims (default)
    all-mpnet-base-v2     Best quality, 768 dims
    BAAI/bge-small-en-v1.5  Good accuracy, 384 dims
        """,
    )

    parser.add_argument(
        "sources",
        nargs="+",
        help="Markdown file(s) or directory containing markdown files",
    )

    # Chunking options
    chunking_group = parser.add_argument_group("Chunking options")
    chunking_group.add_argument(
        "--strategy",
        choices=["fixed", "recursive", "kamradt", "cluster_semantic", "llm_semantic"],
        default="recursive",
        help="Chunking strategy (default: recursive)",
    )
    chunking_group.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Target chunk size in tokens (default: 512)",
    )
    chunking_group.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="Overlap between chunks in tokens (default: 50)",
    )

    # Embedding options
    embed_group = parser.add_argument_group("Embedding options")
    embed_group.add_argument(
        "--provider",
        choices=["sentence-transformers", "openai", "local"],
        default="sentence-transformers",
        help="Embedding provider (default: sentence-transformers, local)",
    )
    embed_group.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="Embedding model (default: all-MiniLM-L6-v2)",
    )
    embed_group.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for embedding (default: 100)",
    )
    embed_group.add_argument(
        "--list-models",
        action="store_true",
        help="List available local embedding models and exit",
    )

    # Vector store options
    store_group = parser.add_argument_group("Vector store options")
    store_group.add_argument(
        "--backend",
        choices=["pgvector", "simple"],
        default="simple",
        help="Vector store backend (default: simple)",
    )
    store_group.add_argument(
        "--collection",
        default="markdown_docs",
        help="Collection/table name (default: markdown_docs)",
    )
    store_group.add_argument(
        "--output-dir",
        default="data/vectordb",
        help="Output directory for simple backend (default: data/vectordb)",
    )
    store_group.add_argument(
        "--pg-connection", help="PostgreSQL connection string for pgvector backend"
    )

    args = parser.parse_args()

    # Handle --list-models
    if args.list_models:
        print("Available local embedding models:")
        print("-" * 40)
        for model, dims in list_local_models().items():
            print(f"  {model:<30} {dims} dims")
        return 0

    print("Vector Database Pipeline")
    print("-" * 30)

    # Collect files
    files = collect_files(args.sources)
    if not files:
        print("No markdown files found in the specified sources.")
        return 1

    print(f"Found {len(files)} markdown file(s)")

    # Map strategy string to enum
    strategy_map = {
        "fixed": ChunkingStrategy.FIXED,
        "recursive": ChunkingStrategy.RECURSIVE,
        "kamradt": ChunkingStrategy.KAMRADT,
        "cluster_semantic": ChunkingStrategy.CLUSTER_SEMANTIC,
        "llm_semantic": ChunkingStrategy.LLM_SEMANTIC,
    }

    # Build configs
    chunking_config = ChunkingConfig(
        strategy=strategy_map[args.strategy],
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap,
    )

    embedding_config = EmbeddingConfig(
        model=args.model,
        batch_size=args.batch_size,
    )

    store_config = VectorStoreConfig(
        collection_name=args.collection,
        persist_directory=args.output_dir,
        pg_connection_string=args.pg_connection,
    )

    # Validate pgvector requirements
    if args.backend == "pgvector" and not args.pg_connection:
        print("Error: --pg-connection required for pgvector backend")
        return 1

    # For semantic chunking strategies, we need an embedding function
    if args.strategy in ["kamradt", "cluster_semantic"]:
        print(f"\nNote: {args.strategy} chunking requires embeddings.")
        print("Setting up embedding function for chunking...")
        from vectordb import create_sentence_transformer_embedding_function

        chunking_config = ChunkingConfig(
            strategy=strategy_map[args.strategy],
            chunk_size=args.chunk_size,
            chunk_overlap=args.overlap,
            embedding_function=create_sentence_transformer_embedding_function(
                args.model
            ),
        )

    # Step 1: Chunking
    print(f"\nStep 1/3: Chunking documents ({args.strategy})...")
    chunks = chunk_markdown(files, chunking_config)
    chunk_stats = get_chunk_statistics(chunks)
    print(f"  Created {len(chunks)} chunks")

    # Step 2: Embeddings
    provider_name = (
        "local"
        if args.provider in ["sentence-transformers", "local"]
        else args.provider
    )
    print(f"\nStep 2/3: Generating embeddings ({provider_name}: {args.model})...")
    embeddings = embed_markdown(chunks, args.provider, embedding_config)
    embedding_stats = get_embedding_statistics(embeddings)
    print(
        f"  Generated {len(embeddings)} embeddings ({embedding_stats['dimensions']} dims)"
    )

    # Step 3: Store
    print(f"\nStep 3/3: Storing in {args.backend}...")
    backend = build_vector_store(embeddings, args.backend, store_config)
    store_stats = get_store_statistics(backend)
    print(f"  Stored {store_stats['total_embeddings']} vectors")

    # Summary
    print_summary(
        files, chunk_stats, embedding_stats, store_stats, args.strategy, args.backend
    )

    if args.backend == "pgvector":
        print(f"\nVector database stored in PostgreSQL table: {args.collection}")
    else:
        print(f"\nVector database saved to: {args.output_dir}")

    return 0


if __name__ == "__main__":
    exit(main())
