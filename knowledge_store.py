"""PostgreSQL/pgvector knowledge-base storage and embedding helpers.

The online workflow only searches this store.  File parsing, chunking and
embedding are handled by ``ingest_knowledge.py`` so that customer decisions do
not pay the ingestion cost on every run.
"""

from __future__ import annotations

import inspect
import json
import os
from collections.abc import Callable, Iterable
from typing import Any

from openai import AsyncOpenAI


def vector_literal(values: Iterable[float]) -> str:
    """Return the pgvector text representation used by parameterized SQL."""

    return "[" + ",".join(str(float(value)) for value in values) + "]"


class EmbeddingClient:
    """Small OpenAI-compatible embedding client.

    The endpoint is configurable because DeepSeek is used for chat in this
    project while an embedding provider may be hosted separately.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str | None = None,
        client: Any | None = None,
    ):
        if not api_key:
            raise ValueError("缺少 Embedding API Key")
        self.model = model
        self.client = client or AsyncOpenAI(
            api_key=api_key,
            base_url=base_url or None,
        )

    async def embed(self, text: str) -> list[float]:
        response = await self.client.embeddings.create(
            model=self.model,
            input=text,
        )
        return list(response.data[0].embedding)


class KnowledgeStore:
    """Async knowledge chunks stored in PostgreSQL with pgvector."""

    def __init__(
        self,
        database_url: str,
        connect: Callable[[], Any] | None = None,
        embedding_dimensions: int = 1536,
    ):
        if embedding_dimensions <= 0:
            raise ValueError("embedding_dimensions 必须为正整数")
        self.database_url = database_url
        self.embedding_dimensions = embedding_dimensions
        self._connect_override = connect

    async def _connect(self):
        if self._connect_override is not None:
            connection = self._connect_override()
            if inspect.isawaitable(connection):
                return await connection
            return connection

        from psycopg import AsyncConnection
        from psycopg.rows import dict_row

        return await AsyncConnection.connect(
            self.database_url,
            autocommit=True,
            row_factory=dict_row,
        )

    async def setup(self) -> None:
        """Create the extension, table and indexes used by retrieval."""

        connection = await self._connect()
        async with connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    "CREATE EXTENSION IF NOT EXISTS vector"
                )
                await cursor.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS knowledge_chunks (
                        id UUID PRIMARY KEY,
                        tenant_id TEXT NOT NULL,
                        corpus_type TEXT NOT NULL,
                        customer_id TEXT,
                        document_id TEXT NOT NULL,
                        chunk_index INTEGER NOT NULL,
                        content TEXT NOT NULL,
                        embedding VECTOR({self.embedding_dimensions}) NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        source_uri TEXT,
                        document_version TEXT,
                        checksum TEXT,
                        active BOOLEAN NOT NULL DEFAULT TRUE,
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        UNIQUE (tenant_id, corpus_type, document_id, chunk_index)
                    )
                    """
                )
                await cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_knowledge_scope
                    ON knowledge_chunks(tenant_id, corpus_type, customer_id)
                    """
                )
                await cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_knowledge_embedding_hnsw
                    ON knowledge_chunks USING hnsw (embedding vector_cosine_ops)
                    """
                )
                await cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_knowledge_content_fts
                    ON knowledge_chunks
                    USING gin (to_tsvector('simple', content))
                    """
                )

    async def upsert_chunks(
        self,
        chunks: Iterable[dict[str, Any]],
    ) -> int:
        """Insert or replace chunks; returns the number of written chunks."""

        rows = list(chunks)
        if not rows:
            return 0

        connection = await self._connect()
        async with connection:
            async with connection.cursor() as cursor:
                documents = {
                    (
                        chunk["tenant_id"],
                        chunk["corpus_type"],
                        chunk["document_id"],
                    )
                    for chunk in rows
                }
                for tenant_id, corpus_type, document_id in documents:
                    await cursor.execute(
                        """
                        UPDATE knowledge_chunks
                        SET active = FALSE, updated_at = NOW()
                        WHERE tenant_id = %s
                          AND corpus_type = %s
                          AND document_id = %s
                        """,
                        (tenant_id, corpus_type, document_id),
                    )
                for chunk in rows:
                    metadata = json.dumps(
                        chunk.get("metadata") or {},
                        ensure_ascii=False,
                    )
                    await cursor.execute(
                        """
                        INSERT INTO knowledge_chunks (
                            id, tenant_id, corpus_type, customer_id,
                            document_id, chunk_index, content, embedding,
                            metadata, source_uri, document_version, checksum,
                            active, updated_at
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s::vector,
                            %s::jsonb, %s, %s, %s, %s, NOW()
                        )
                        ON CONFLICT (
                            tenant_id, corpus_type, document_id, chunk_index
                        )
                        DO UPDATE SET
                            corpus_type = EXCLUDED.corpus_type,
                            customer_id = EXCLUDED.customer_id,
                            content = EXCLUDED.content,
                            embedding = EXCLUDED.embedding,
                            metadata = EXCLUDED.metadata,
                            source_uri = EXCLUDED.source_uri,
                            document_version = EXCLUDED.document_version,
                            checksum = EXCLUDED.checksum,
                            active = EXCLUDED.active,
                            updated_at = NOW()
                        """,
                        (
                            chunk["id"],
                            chunk["tenant_id"],
                            chunk["corpus_type"],
                            chunk.get("customer_id"),
                            chunk["document_id"],
                            chunk["chunk_index"],
                            chunk["content"],
                            vector_literal(chunk["embedding"]),
                            metadata,
                            chunk.get("source_uri"),
                            chunk.get("document_version"),
                            chunk.get("checksum"),
                            chunk.get("active", True),
                        ),
                    )
        return len(rows)

    async def search(
        self,
        *,
        tenant_id: str,
        corpus_type: str,
        query: str,
        embedding: Iterable[float],
        customer_id: str | None = None,
        limit: int = 5,
        strict_customer: bool = False,
    ) -> list[dict[str, Any]]:
        """Run scoped vector + lightweight full-text hybrid retrieval."""

        if limit <= 0:
            return []

        customer_clause = (
            "customer_id = %s"
            if strict_customer
            else "(customer_id IS NULL OR customer_id = %s)"
        )
        parameters = (
            vector_literal(embedding),
            query,
            tenant_id,
            corpus_type,
            customer_id,
            limit,
        )

        connection = await self._connect()
        async with connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    f"""
                    SELECT id, customer_id, document_id, chunk_index,
                           content, metadata, source_uri, document_version,
                           checksum,
                           (
                               0.85 * (1 - (embedding <=> %s::vector))
                               + 0.15 * ts_rank(
                                   to_tsvector('simple', content),
                                   plainto_tsquery('simple', %s)
                               )
                           ) AS score
                    FROM knowledge_chunks
                    WHERE tenant_id = %s
                      AND corpus_type = %s
                      AND active = TRUE
                      AND {customer_clause}
                    ORDER BY score DESC
                    LIMIT %s
                    """,
                    parameters,
                )
                rows = await cursor.fetchall()

        return [
            {
                "id": str(row["id"]),
                "customer_id": row["customer_id"],
                "document_id": row["document_id"],
                "chunk_index": row["chunk_index"],
                "content": row["content"],
                "metadata": (
                    json.loads(row["metadata"])
                    if isinstance(row["metadata"], str)
                    else row["metadata"]
                ),
                "source_uri": row["source_uri"],
                "document_version": row["document_version"],
                "checksum": row["checksum"],
                "score": float(row["score"]),
            }
            for row in rows
        ]

    async def table_ready(self) -> bool:
        """检查知识库表是否已创建。只读，不会建表。"""

        connection = await self._connect()
        async with connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """
                    SELECT to_regclass('public.knowledge_chunks') AS table_name
                    """
                )
                row = await cursor.fetchone()

        return bool(row and row["table_name"])

    async def corpus_stats(self, tenant_id: str) -> dict[str, int]:
        """只读统计各语料在租户下已启用的片段数。"""

        connection = await self._connect()
        async with connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """
                    SELECT corpus_type, COUNT(*) AS total
                    FROM knowledge_chunks
                    WHERE tenant_id = %s AND active = TRUE
                    GROUP BY corpus_type
                    """,
                    (tenant_id,),
                )
                rows = await cursor.fetchall()

        return {
            row["corpus_type"]: int(row["total"])
            for row in rows
        }


def knowledge_enabled() -> bool:
    return os.getenv("KNOWLEDGE_BASE_ENABLED", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
