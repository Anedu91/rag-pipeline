#!/usr/bin/env python3
"""
RAG Query CLI - Ask questions to your knowledge base

Usage:
    # Single question
    python rag_query.py "What is machine learning?"

    # Interactive mode
    python rag_query.py --interactive

    # Configure backend and model
    python rag_query.py --backend simple --llm ollama --model llama3.2

    # Use specific collection
    python rag_query.py --collection my_docs "What is Python?"
"""

import argparse
import sys
from pathlib import Path

from vectordb import (
    EmbeddingConfig,
    RAGConfig,
    VectorStoreConfig,
    create_embed_function,
    format_response_with_sources,
    get_backend,
    rag_pipeline,
)


def print_banner():
    """Print welcome banner."""
    print("\n" + "=" * 80)
    print(" " * 25 + "RAG QUERY - Ask Your Knowledge Base")
    print("=" * 80 + "\n")


def print_help_text():
    """Print helpful information."""
    print("💡 Tips:")
    print("  - Type 'exit' or 'quit' to end interactive mode")
    print("  - Type 'help' to see this message again")
    print("  - Type 'stats' to see knowledge base statistics")
    print("  - Press Ctrl+C to exit")
    print()


def get_statistics(backend):
    """Get and display knowledge base statistics."""
    from vectordb.vector_store import get_store_statistics

    stats = get_store_statistics(backend)
    print("\n" + "=" * 80)
    print("KNOWLEDGE BASE STATISTICS")
    print("=" * 80)
    print(f"Total embeddings: {stats['total_embeddings']}")
    print("=" * 80 + "\n")


def ask_question(
    question: str,
    backend,
    embed_fn,
    provider_type: str,
    rag_config: RAGConfig,
    show_sources: bool = True,
    verbose: bool = False,
):
    """Ask a question and get an answer."""
    if verbose:
        print(f"\n🔍 Searching knowledge base...")

    try:
        response = rag_pipeline(
            question=question,
            backend=backend,
            embed_fn=embed_fn,
            provider_type=provider_type,
            config=rag_config,
        )
        # Display answer
        print("\n" + "=" * 80)
        print("ANSWER:")
        print("=" * 80)
        print(response.answer)
        print()

        # Display sources if requested
        if show_sources and response.sources:
            print("=" * 80)
            print("SOURCES:")
            print("=" * 80)
            for i, result in enumerate(response.sources[:5], 1):
                file_name = result.chunk.metadata.get(
                    "file_name", result.chunk.source_file
                )
                print(f"\n[{i}] {file_name} (Relevance: {result.score:.3f})")
                print(f"    {result.chunk.content[:150]}...")

        # Display metadata if verbose
        if verbose:
            print("\n" + "=" * 80)
            print("METADATA:")
            print("=" * 80)
            print(f"Model: {response.model}")
            if response.tokens_used:
                print(f"Tokens used: {response.tokens_used}")
            print(f"Sources retrieved: {len(response.sources)}")

        print("\n" + "=" * 80 + "\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        if "ollama" in str(e).lower() or provider_type == "ollama":
            print("  - Make sure Ollama is running: ollama serve")
            print("  - Check if model is pulled: ollama pull llama3.2")
            print("  - Try a different model: --model mistral")
        elif "openai" in str(e).lower() or provider_type == "openai":
            print("  - Check your OPENAI_API_KEY environment variable")
            print("  - Verify you have API credits")
        elif "No embeddings found" in str(e) or "count" in str(e).lower():
            print("  - Your knowledge base is empty!")
            print("  - Build it first using: python vectordb_main.py <files>")
        else:
            print(f"  - Error details: {e}")
        print()


def interactive_mode(backend, embed_fn, provider_type: str, rag_config: RAGConfig):
    """Run interactive question-answering mode."""
    print_banner()
    print("🤖 Interactive Mode - Ask questions to your knowledge base\n")
    print_help_text()

    while True:
        try:
            # Get question from user
            question = input("\n💬 You: ").strip()

            # Handle special commands
            if not question:
                continue

            if question.lower() in ["exit", "quit", "q"]:
                print("\n👋 Goodbye!\n")
                break

            if question.lower() == "help":
                print_help_text()
                continue

            if question.lower() == "stats":
                get_statistics(backend)
                continue

            # Ask question
            ask_question(
                question,
                backend,
                embed_fn,
                provider_type,
                rag_config,
                show_sources=True,
            )

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!\n")
            break
        except EOFError:
            print("\n\n👋 Goodbye!\n")
            break


def main():
    parser = argparse.ArgumentParser(
        description="Ask questions to your RAG knowledge base",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single question
  python rag_query.py "What is machine learning?"

  # Interactive mode
  python rag_query.py --interactive

  # Use Ollama with specific model
  python rag_query.py --llm ollama --model llama3.2 --interactive

  # Use OpenAI
  python rag_query.py --llm openai --model gpt-3.5-turbo "Explain RAG"

  # Specify collection
  python rag_query.py --collection my_docs --interactive

  # Verbose output
  python rag_query.py --verbose "What is Python?"
        """,
    )

    # Question argument (positional, optional)
    parser.add_argument(
        "question", nargs="?", help="Question to ask (omit for interactive mode)"
    )

    # Mode
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Interactive mode (continuous Q&A)",
    )

    # Vector store options
    store_group = parser.add_argument_group("Vector store options")
    store_group.add_argument(
        "--backend",
        choices=["simple", "pgvector"],
        default="simple",
        help="Vector store backend (default: simple)",
    )
    store_group.add_argument(
        "--collection",
        default="default",
        help="Collection name (default: default)",
    )
    store_group.add_argument(
        "--vector-dir",
        default="data/vectordb",
        help="Vector store directory for simple backend (default: data/vectordb)",
    )
    store_group.add_argument(
        "--pg-connection", help="PostgreSQL connection string for pgvector backend"
    )

    # Embedding options
    embed_group = parser.add_argument_group("Embedding options")
    embed_group.add_argument(
        "--embed-model",
        default="all-MiniLM-L6-v2",
        help="Embedding model (default: all-MiniLM-L6-v2)",
    )

    # LLM options
    llm_group = parser.add_argument_group("LLM options")
    llm_group.add_argument(
        "--llm",
        choices=["ollama", "openai", "anthropic", "groq"],
        default="ollama",
        help="LLM provider (default: ollama)",
    )
    llm_group.add_argument(
        "--model",
        default="llama3.2",
        help="LLM model name (default: llama3.2 for ollama, gpt-3.5-turbo for openai)",
    )
    llm_group.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="LLM temperature (default: 0.7)",
    )
    llm_group.add_argument(
        "--max-tokens",
        type=int,
        default=1000,
        help="Maximum tokens in response (default: 1000)",
    )
    llm_group.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve (default: 5)",
    )

    # Output options
    output_group = parser.add_argument_group("Output options")
    output_group.add_argument(
        "--no-sources", action="store_true", help="Don't show sources"
    )
    output_group.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose output with metadata"
    )

    args = parser.parse_args()

    # Validate: need either question or interactive mode
    if not args.question and not args.interactive:
        parser.print_help()
        print("\n❌ Error: Provide a question or use --interactive mode\n")
        sys.exit(1)

    # Set default model based on provider
    if args.model == "llama3.2" and args.llm != "ollama":
        if args.llm == "openai":
            args.model = "gpt-3.5-turbo"
        elif args.llm == "anthropic":
            args.model = "claude-3-5-sonnet-20241022"
        elif args.llm == "groq":
            args.model = "llama-3.1-70b-versatile"

    # Setup vector store
    print(f"📚 Loading knowledge base from '{args.collection}'...")
    vector_config = VectorStoreConfig(
        collection_name=args.collection,
        persist_directory=args.vector_dir,
        pg_connection_string=args.pg_connection,
    )

    try:
        backend = get_backend(args.backend, vector_config)
        from vectordb.vector_store import get_store_statistics

        stats = get_store_statistics(backend)

        if stats["total_embeddings"] == 0:
            print("\n❌ Error: Knowledge base is empty!")
            print("\nBuild your knowledge base first using vectordb_main.py:")
            print("  python vectordb_main.py <your_files.md>\n")
            sys.exit(1)

        print(f"✓ Loaded {stats['total_embeddings']} embeddings\n")

    except Exception as e:
        print(f"\n❌ Error loading knowledge base: {e}")
        print("\nMake sure you've built your knowledge base first:")
        print("  python vectordb_main.py <your_files.md>\n")
        sys.exit(1)

    # Setup embedding function
    print(f"🔧 Initializing embeddings model ({args.embed_model})...")
    embed_config = EmbeddingConfig(model=args.embed_model)
    embed_fn = create_embed_function("sentence-transformers", embed_config)
    print("✓ Ready\n")

    # Setup RAG config
    print(f"🤖 Initializing LLM ({args.llm} / {args.model})...")
    rag_config = RAGConfig(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_k=args.top_k,
    )
    print("✓ Ready\n")

    # Run query mode
    if args.interactive:
        interactive_mode(backend, embed_fn, args.llm, rag_config)
    else:
        # Single question mode
        print_banner()
        print(f"❓ Question: {args.question}\n")
        ask_question(
            args.question,
            backend,
            embed_fn,
            args.llm,
            rag_config,
            show_sources=not args.no_sources,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    main()
