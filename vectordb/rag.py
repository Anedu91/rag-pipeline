"""
RAG (Retrieval Augmented Generation) module.

Pure functional approach - combines vector search with LLM generation.
Supports multiple LLM providers for flexibility.

Available providers:
- openai: OpenAI models (GPT-4, GPT-3.5-turbo, etc.)
- anthropic: Claude models (Claude 3.5 Sonnet, etc.)
- ollama: Local models via Ollama
- groq: Fast inference via Groq API

Typical RAG Flow:
1. User asks a question
2. Convert question to embedding
3. Search vector DB for relevant chunks
4. Format chunks as context
5. Send context + question to LLM
6. Return LLM's answer with sources
"""

import os
from dataclasses import dataclass
from typing import Callable, Protocol

from dotenv import load_dotenv

from vectordb.vector_store import SearchResult, VectorStoreBackend, query_vector_store

load_dotenv()


@dataclass
class RAGConfig:
    """
    Configuration for RAG pipeline.

    Attributes:
        model: LLM model name
        temperature: Sampling temperature (0.0-1.0)
        max_tokens: Maximum tokens in response
        top_k: Number of chunks to retrieve
        context_max_length: Maximum context length in characters
        include_sources: Whether to include source citations
        system_prompt: System prompt for the LLM
        api_key: API key for cloud providers
    """

    model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 1000
    top_k: int = 5
    context_max_length: int = 4000
    include_sources: bool = True
    system_prompt: str | None = None
    api_key: str | None = None


@dataclass(frozen=True)
class RAGResponse:
    """Immutable RAG response with answer and metadata."""

    answer: str
    sources: list[SearchResult]
    context_used: str
    model: str
    tokens_used: int | None = None


class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    def generate(
        self, prompt: str, system_prompt: str | None = None
    ) -> tuple[str, int | None]:
        """
        Generate text from prompt.

        Returns:
            (generated_text, tokens_used)
        """
        ...

    @property
    def model_name(self) -> str:
        """Return the model name."""
        ...


def create_openai_provider(config: RAGConfig) -> LLMProvider:
    """Create an OpenAI LLM provider."""
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
        def generate(
            self, prompt: str, system_prompt: str | None = None
        ) -> tuple[str, int | None]:
            messages = []

            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            messages.append({"role": "user", "content": prompt})

            response = client.chat.completions.create(
                model=config.model,
                messages=messages,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )

            answer = response.choices[0].message.content or ""
            tokens = response.usage.total_tokens if response.usage else None

            return answer, tokens

        @property
        def model_name(self) -> str:
            return config.model

    return OpenAIProvider()


def create_anthropic_provider(config: RAGConfig) -> LLMProvider:
    """Create an Anthropic Claude LLM provider."""
    try:
        from anthropic import Anthropic
    except ImportError:
        raise ImportError("anthropic package required. Install with: uv add anthropic")

    api_key = config.api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "Anthropic API key required. Set ANTHROPIC_API_KEY env var or pass in config."
        )

    client = Anthropic(api_key=api_key)

    class AnthropicProvider:
        def generate(
            self, prompt: str, system_prompt: str | None = None
        ) -> tuple[str, int | None]:
            response = client.messages.create(
                model=config.model,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                system=system_prompt or "You are a helpful assistant.",
                messages=[{"role": "user", "content": prompt}],
            )

            answer = response.content[0].text if response.content else ""
            tokens = response.usage.input_tokens + response.usage.output_tokens

            return answer, tokens

        @property
        def model_name(self) -> str:
            return config.model

    return AnthropicProvider()


def create_ollama_provider(config: RAGConfig) -> LLMProvider:
    """
    Create a local Ollama LLM provider.

    Requires Ollama to be installed and running locally.
    Install: https://ollama.ai/
    """
    try:
        import ollama
    except ImportError:
        raise ImportError("ollama package required. Install with: uv add ollama-python")

    class OllamaProvider:
        def generate(
            self, prompt: str, system_prompt: str | None = None
        ) -> tuple[str, int | None]:
            messages = []

            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            messages.append({"role": "user", "content": prompt})

            ,l.
            J = ollama.chat(
           J = ollama.chat(
                model=config.model,
                messages=messages,
                options={
                    "temperature": config.temperature,
                    "num_predict": config.max_tokens,
                },
            )

            answer = response["message"]["content"]
            return answer, None

        @property
        def model_name(self) -> str:
            return config.model

    return OllamaProvider()


def create_groq_provider(config: RAGConfig) -> LLMProvider:
    """Create a Groq LLM provider (fast inference)."""
    try:
        from groq import Groq
    except ImportError:
        raise ImportError("groq package required. Install with: uv add groq")

    api_key = config.api_key or os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "Groq API key required. Set GROQ_API_KEY env var or pass in config."
        )

    client = Groq(api_key=api_key)

    class GroqProvider:
        def generate(
            self, prompt: str, system_prompt: str | None = None
        ) -> tuple[str, int | None]:
            messages = []

            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            messages.append({"role": "user", "content": prompt})

            response = client.chat.completions.create(
                model=config.model,
                messages=messages,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )

            answer = response.choices[0].message.content or ""
            tokens = response.usage.total_tokens if response.usage else None

            return answer, tokens

        @property
        def model_name(self) -> str:
            return config.model

    return GroqProvider()


def get_llm_provider(
    provider_type: str = "openai", config: RAGConfig | None = None
) -> LLMProvider:
    """
    Get an LLM provider by type.

    Args:
        provider_type: "openai", "anthropic", "ollama", or "groq"
        config: RAG configuration

    Returns:
        An LLMProvider instance
    """
    config = config or RAGConfig()

    providers = {
        "openai": create_openai_provider,
        "anthropic": create_anthropic_provider,
        "ollama": create_ollama_provider,
        "groq": create_groq_provider,
    }

    if provider_type not in providers:
        raise ValueError(
            f"Unknown provider: {provider_type}. Available: {list(providers.keys())}"
        )

    return providers[provider_type](config)


def format_context(search_results: list[SearchResult], max_length: int = 4000) -> str:
    """
    Format search results into context string for LLM.

    Args:
        search_results: List of search results from vector store
        max_length: Maximum length of context in characters

    Returns:
        Formatted context string
    """
    if not search_results:
        return "No relevant context found."

    context_parts = []
    current_length = 0

    for i, result in enumerate(search_results, 1):
        chunk_text = f"[Source {i}: {result.chunk.metadata.get('file_name', result.chunk.source_file)} (Score: {result.score:.3f})]\n{result.chunk.content}\n"

        if current_length + len(chunk_text) > max_length:
            break

        context_parts.append(chunk_text)
        current_length += len(chunk_text)

    return "\n---\n".join(context_parts)


def build_prompt(question: str, context: str, include_instructions: bool = True) -> str:
    """
    Build the final prompt for the LLM.

    Args:
        question: User's question
        context: Retrieved context
        include_instructions: Whether to include answer instructions

    Returns:
        Complete prompt string
    """
    prompt_parts = []

    if include_instructions:
        prompt_parts.append(
            "Answer the question based on the context provided below. "
            "If the context doesn't contain enough information to answer the question, "
            "say so clearly. Do not make up information.\n"
        )

    prompt_parts.append(f"Context:\n{context}\n")
    prompt_parts.append(f"Question: {question}\n")
    prompt_parts.append("Answer:")

    return "\n".join(prompt_parts)


def retrieve_context(
    query: str,
    backend: VectorStoreBackend,
    embed_fn: Callable[[str], list[float]],
    top_k: int = 5,
) -> list[SearchResult]:
    """
    Retrieve relevant context from vector store.

    Args:
        query: User's question
        backend: Vector store backend
        embed_fn: Function to embed the query
        top_k: Number of chunks to retrieve

    Returns:
        List of search results
    """
    return query_vector_store(query, backend, embed_fn, top_k)


def generate_answer(
    question: str,
    context: str,
    provider: LLMProvider,
    system_prompt: str | None = None,
) -> tuple[str, int | None]:
    """
    Generate answer using LLM.

    Args:
        question: User's question
        context: Retrieved context
        provider: LLM provider
        system_prompt: Optional system prompt

    Returns:
        (answer, tokens_used)
    """
    prompt = build_prompt(question, context)
    return provider.generate(prompt, system_prompt)


def rag_query(
    question: str,
    backend: VectorStoreBackend,
    embed_fn: Callable[[str], list[float]],
    provider: LLMProvider,
    config: RAGConfig | None = None,
) -> RAGResponse:
    """
    Complete RAG pipeline: retrieve context and generate answer.

    Args:
        question: User's question
        backend: Vector store backend
        embed_fn: Function to embed queries
        provider: LLM provider
        config: RAG configuration

    Returns:
        RAGResponse with answer and metadata
    """
    config = config or RAGConfig()

    # Step 1: Retrieve relevant context
    search_results = retrieve_context(question, backend, embed_fn, config.top_k)

    # Step 2: Format context
    context = format_context(search_results, config.context_max_length)

    # Step 3: Generate answer
    answer, tokens = generate_answer(question, context, provider, config.system_prompt)

    # Step 4: Build response
    return RAGResponse(
        answer=answer,
        sources=search_results,
        context_used=context,
        model=provider.model_name,
        tokens_used=tokens,
    )


def rag_pipeline(
    question: str,
    backend: VectorStoreBackend,
    embed_fn: Callable[[str], list[float]],
    provider_type: str = "openai",
    config: RAGConfig | None = None,
) -> RAGResponse:
    """
    Convenience function for complete RAG pipeline.

    Args:
        question: User's question
        backend: Vector store backend
        embed_fn: Function to embed queries
        provider_type: LLM provider type ("openai", "anthropic", "ollama", "groq")
        config: RAG configuration

    Returns:
        RAGResponse with answer and metadata

    Example:
        from vectordb.rag import rag_pipeline
        from vectordb.embeddings import create_embed_function
        from vectordb.vector_store import get_backend, VectorStoreConfig

        # Setup
        backend = get_backend("chromadb", VectorStoreConfig())
        embed_fn = create_embed_function()

        # Query
        response = rag_pipeline(
            question="What is machine learning?",
            backend=backend,
            embed_fn=embed_fn,
            provider_type="openai"
        )

        print(response.answer)
    """
    config = config or RAGConfig()
    provider = get_llm_provider(provider_type, config)

    return rag_query(question, backend, embed_fn, provider, config)


def format_response_with_sources(response: RAGResponse) -> str:
    """
    Format RAG response with sources for display.

    Args:
        response: RAG response object

    Returns:
        Formatted string with answer and sources
    """
    output_parts = []

    # Answer
    output_parts.append("=" * 80)
    output_parts.append("ANSWER:")
    output_parts.append("=" * 80)
    output_parts.append(response.answer)
    output_parts.append("")

    # Sources
    if response.sources:
        output_parts.append("=" * 80)
        output_parts.append("SOURCES:")
        output_parts.append("=" * 80)

        for i, result in enumerate(response.sources, 1):
            file_name = result.chunk.metadata.get("file_name", result.chunk.source_file)
            output_parts.append(f"\n[{i}] {file_name} (Relevance: {result.score:.3f})")
            output_parts.append(f"    {result.chunk.content[:150]}...")

    # Metadata
    output_parts.append("\n" + "=" * 80)
    output_parts.append("METADATA:")
    output_parts.append("=" * 80)
    output_parts.append(f"Model: {response.model}")
    if response.tokens_used:
        output_parts.append(f"Tokens used: {response.tokens_used}")
    output_parts.append(f"Sources retrieved: {len(response.sources)}")

    return "\n".join(output_parts)


def batch_rag_query(
    questions: list[str],
    backend: VectorStoreBackend,
    embed_fn: Callable[[str], list[float]],
    provider: LLMProvider,
    config: RAGConfig | None = None,
) -> list[RAGResponse]:
    """
    Process multiple questions through RAG pipeline.

    Args:
        questions: List of questions
        backend: Vector store backend
        embed_fn: Function to embed queries
        provider: LLM provider
        config: RAG configuration

    Returns:
        List of RAG responses
    """
    return [
        rag_query(question, backend, embed_fn, provider, config)
        for question in questions
    ]
