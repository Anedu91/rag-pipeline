"""
Vector store module for storing and querying embeddings.

Pure functional approach - all functions are stateless.
Supports multiple vector database backends.

Available backends:
- simple: JSON file-based (no dependencies, good for development)
- pgvector: PostgreSQL with pgvector extension (production-ready)
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from dotenv import load_dotenv

from vectordb.chunking import Chunk
from vectordb.embeddings import Embedding

# Load environment variables
load_dotenv()


@dataclass(frozen=True)
class SearchResult:
    """Immutable search result with score and metadata."""

    chunk: Chunk
    score: float
    embedding_id: str


@dataclass
class VectorStoreConfig:
    """
    Configuration for vector store.

    Attributes:
        collection_name: Name of the collection/table
        persist_directory: Directory for file-based stores
        distance_metric: Distance metric (cosine, l2, inner_product)

        # PostgreSQL/pgvector settings
        pg_connection_string: PostgreSQL connection string
        pg_table_name: Table name for pgvector (defaults to collection_name)
    """

    collection_name: str = "default"
    persist_directory: str = "data/vectordb"
    distance_metric: str = "cosine"

    # PostgreSQL settings
    pg_connection_string: str | None = None
    pg_table_name: str | None = None


class VectorStoreBackend(Protocol):
    """Protocol for vector store backends."""

    def add_embeddings(self, embeddings: list[Embedding]) -> list[str]:
        """Add embeddings to the store. Returns list of IDs."""
        ...

    def search(
        self, query_vector: list[float], top_k: int
    ) -> list[tuple[str, float, dict]]:
        """Search for similar vectors. Returns (id, score, metadata)."""
        ...

    def delete(self, ids: list[str]) -> None:
        """Delete embeddings by ID."""
        ...

    def count(self) -> int:
        """Return total number of embeddings."""
        ...

    def persist(self) -> None:
        """Persist the store to disk."""
        ...


def create_simple_backend(config: VectorStoreConfig) -> VectorStoreBackend:
    """Create a simple JSON-based vector store backend (no dependencies)."""

    persist_dir = Path(config.persist_directory)
    persist_dir.mkdir(parents=True, exist_ok=True)
    store_path = persist_dir / f"{config.collection_name}.json"

    if store_path.exists():
        with open(store_path, "r") as f:
            store_data = json.load(f)
    else:
        store_data = {"embeddings": {}, "metadata": {}}

    def cosine_similarity(v1: list[float], v2: list[float]) -> float:
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = sum(a * a for a in v1) ** 0.5
        norm2 = sum(b * b for b in v2) ** 0.5
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    class SimpleBackend:
        def add_embeddings(self, embeddings: list[Embedding]) -> list[str]:
            ids = []
            for emb in embeddings:
                emb_id = f"{emb.chunk.source_file}_{emb.chunk.chunk_index}"
                store_data["embeddings"][emb_id] = list(emb.vector)
                store_data["metadata"][emb_id] = {
                    "source_file": emb.chunk.source_file,
                    "chunk_index": emb.chunk.chunk_index,
                    "start_char": emb.chunk.start_char,
                    "end_char": emb.chunk.end_char,
                    "content": emb.chunk.content,
                    "model": emb.model,
                    "file_name": emb.chunk.metadata.get("file_name", ""),
                }
                ids.append(emb_id)
            return ids

        def search(
            self, query_vector: list[float], top_k: int
        ) -> list[tuple[str, float, dict]]:
            scores = []
            for emb_id, vector in store_data["embeddings"].items():
                score = cosine_similarity(query_vector, vector)
                metadata = store_data["metadata"].get(emb_id, {})
                scores.append((emb_id, score, metadata))

            scores.sort(key=lambda x: x[1], reverse=True)
            return scores[:top_k]

        def delete(self, ids: list[str]) -> None:
            for id_ in ids:
                store_data["embeddings"].pop(id_, None)
                store_data["metadata"].pop(id_, None)

        def count(self) -> int:
            return len(store_data["embeddings"])

        def persist(self) -> None:
            with open(store_path, "w") as f:
                json.dump(store_data, f)

    return SimpleBackend()


def create_pgvector_backend(config: VectorStoreConfig) -> VectorStoreBackend:
    """
    Create a PostgreSQL pgvector backend.

    Requires:
    - PostgreSQL with pgvector extension installed
    - psycopg2 or psycopg package

    Connection string format:
        postgresql://user:password@host:port/database

    Can be provided via:
    - config.pg_connection_string (direct connection string)
    - POSTGRES_CONNECTION_STRING env var
    - Individual env vars: POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_HOST,
      POSTGRES_PORT, POSTGRES_DB
    """
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError:
        try:
            import psycopg2 as psycopg
            from psycopg2.extras import RealDictCursor

            dict_row = RealDictCursor
        except ImportError:
            raise ImportError(
                "psycopg or psycopg2 required for pgvector. "
                "Install with: uv add psycopg[binary] or uv add psycopg2-binary"
            )

    # Build connection string from config or environment variables
    connection_string = config.pg_connection_string

    if not connection_string:
        # Try POSTGRES_CONNECTION_STRING env var
        connection_string = os.getenv("POSTGRES_CONNECTION_STRING")

    if not connection_string:
        # Build from individual components
        user = os.getenv("POSTGRES_USER")
        password = os.getenv("POSTGRES_PASSWORD")
        host = os.getenv("POSTGRES_HOST", "localhost")
        port = os.getenv("POSTGRES_PORT", "5432")
        database = os.getenv("POSTGRES_DB")

        if user and password and database:
            connection_string = (
                f"postgresql://{user}:{password}@{host}:{port}/{database}"
            )
        else:
            raise ValueError(
                "PostgreSQL connection required. Provide either:\n"
                "  1. pg_connection_string in VectorStoreConfig, or\n"
                "  2. POSTGRES_CONNECTION_STRING env var, or\n"
                "  3. POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB env vars"
            )

    table_name = config.pg_table_name or config.collection_name
    table_name = table_name.replace("-", "_").replace(" ", "_")

    distance_ops = {
        "cosine": "<=>",
        "l2": "<->",
        "inner_product": "<#>",
    }
    distance_op = distance_ops.get(config.distance_metric, "<=>")

    conn = psycopg.connect(connection_string)

    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.commit()

    class PgVectorBackend:
        def _ensure_table(self, dimensions: int) -> None:
            with conn.cursor() as cur:
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        id TEXT PRIMARY KEY,
                        embedding vector({dimensions}),
                        content TEXT,
                        source_file TEXT,
                        chunk_index INTEGER,
                        start_char INTEGER,
                        end_char INTEGER,
                        file_name TEXT,
                        model TEXT,
                        metadata JSONB
                    )
                """)
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS {table_name}_embedding_idx
                    ON {table_name}
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100)
                """)
                conn.commit()

        def add_embeddings(self, embeddings: list[Embedding]) -> list[str]:
            if not embeddings:
                return []

            dimensions = len(embeddings[0].vector)
            self._ensure_table(dimensions)

            ids = []
            with conn.cursor() as cur:
                for emb in embeddings:
                    emb_id = f"{emb.chunk.source_file}_{emb.chunk.chunk_index}"
                    vector_str = "[" + ",".join(str(v) for v in emb.vector) + "]"

                    cur.execute(
                        f"""
                        INSERT INTO {table_name}
                        (id, embedding, content, source_file, chunk_index,
                         start_char, end_char, file_name, model, metadata)
                        VALUES (%s, %s::vector, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO UPDATE SET
                            embedding = EXCLUDED.embedding,
                            content = EXCLUDED.content,
                            metadata = EXCLUDED.metadata
                    """,
                        (
                            emb_id,
                            vector_str,
                            emb.chunk.content,
                            emb.chunk.source_file,
                            emb.chunk.chunk_index,
                            emb.chunk.start_char,
                            emb.chunk.end_char,
                            emb.chunk.metadata.get("file_name", ""),
                            emb.model,
                            json.dumps(emb.chunk.metadata),
                        ),
                    )
                    ids.append(emb_id)
                conn.commit()

            return ids

        def search(
            self, query_vector: list[float], top_k: int
        ) -> list[tuple[str, float, dict]]:
            vector_str = "[" + ",".join(str(v) for v in query_vector) + "]"

            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    f"""
                    SELECT
                        id,
                        content,
                        source_file,
                        chunk_index,
                        start_char,
                        end_char,
                        file_name,
                        model,
                        1 - (embedding {distance_op} %s::vector) as score
                    FROM {table_name}
                    ORDER BY embedding {distance_op} %s::vector
                    LIMIT %s
                """,
                    (vector_str, vector_str, top_k),
                )

                results = []
                for row in cur.fetchall():
                    metadata = {
                        "content": row["content"],
                        "source_file": row["source_file"],
                        "chunk_index": row["chunk_index"],
                        "start_char": row["start_char"],
                        "end_char": row["end_char"],
                        "file_name": row["file_name"],
                        "model": row["model"],
                    }
                    results.append((row["id"], row["score"], metadata))

                return results

        def delete(self, ids: list[str]) -> None:
            if not ids:
                return
            with conn.cursor() as cur:
                cur.execute(f"DELETE FROM {table_name} WHERE id = ANY(%s)", (ids,))
                conn.commit()

        def count(self) -> int:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {table_name}")
                result = cur.fetchone()
                return result[0] if result else 0

        def persist(self) -> None:
            conn.commit()

    return PgVectorBackend()


def get_backend(backend_type: str, config: VectorStoreConfig) -> VectorStoreBackend:
    """
    Get a vector store backend by type.

    Args:
        backend_type: "pgvector" or "simple"
        config: Vector store configuration

    Returns:
        A VectorStoreBackend instance
    """
    backends = {
        "pgvector": create_pgvector_backend,
        "simple": create_simple_backend,
    }

    if backend_type not in backends:
        raise ValueError(
            f"Unknown backend: {backend_type}. Available: {list(backends.keys())}"
        )

    return backends[backend_type](config)


def store_embeddings(
    embeddings: list[Embedding], backend: VectorStoreBackend
) -> list[str]:
    """
    Store embeddings in the vector store.

    Args:
        embeddings: List of Embedding objects
        backend: Vector store backend

    Returns:
        List of embedding IDs
    """
    ids = backend.add_embeddings(embeddings)
    backend.persist()
    return ids


def search_similar(
    query_vector: list[float], backend: VectorStoreBackend, top_k: int = 10
) -> list[SearchResult]:
    """
    Search for similar embeddings.

    Args:
        query_vector: Query embedding vector
        backend: Vector store backend
        top_k: Number of results to return

    Returns:
        List of SearchResult objects
    """
    results = backend.search(query_vector, top_k)

    search_results = []
    for emb_id, score, metadata in results:
        chunk = Chunk(
            content=metadata.get("content", metadata.get("document", "")),
            source_file=metadata.get("source_file", ""),
            chunk_index=metadata.get("chunk_index", 0),
            start_char=metadata.get("start_char", 0),
            end_char=metadata.get("end_char", 0),
            metadata={"file_name": metadata.get("file_name", "")},
        )
        search_results.append(
            SearchResult(chunk=chunk, score=score, embedding_id=emb_id)
        )

    return search_results


def delete_embeddings(ids: list[str], backend: VectorStoreBackend) -> None:
    """Delete embeddings by ID."""
    backend.delete(ids)
    backend.persist()


def get_store_statistics(backend: VectorStoreBackend) -> dict:
    """Get statistics about the vector store."""
    return {
        "total_embeddings": backend.count(),
    }


def build_vector_store(
    embeddings: list[Embedding],
    backend_type: str = "simple",
    config: VectorStoreConfig | None = None,
) -> VectorStoreBackend:
    """
    Main entry point for building a vector store from embeddings.

    Args:
        embeddings: List of Embedding objects
        backend_type: "simple" or "pgvector"
        config: Vector store configuration

    Returns:
        Configured VectorStoreBackend with embeddings stored
    """
    config = config or VectorStoreConfig()
    backend = get_backend(backend_type, config)
    store_embeddings(embeddings, backend)
    return backend


def query_vector_store(
    query_text: str,
    backend: VectorStoreBackend,
    embed_fn: Callable[[str], list[float]],
    top_k: int = 10,
) -> list[SearchResult]:
    """
    Query the vector store with text.

    Args:
        query_text: Text to search for
        backend: Vector store backend
        embed_fn: Function to convert text to embedding
        top_k: Number of results to return

    Returns:
        List of SearchResult objects
    """
    query_vector = embed_fn(query_text)
    return search_similar(query_vector, backend, top_k)
