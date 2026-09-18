import json
import inspect
from collections.abc import Callable
from typing import Any


class SalesMemory:
    """使用 PostgreSQL、按 customer_id 隔离的长期记忆存储。"""

    def __init__(
        self,
        database_url: str,
        connect: Callable[[], Any] | None = None,
    ):
        self.database_url = database_url
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
        connection = await self._connect()
        async with connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS sales_memories (
                        id BIGSERIAL PRIMARY KEY,
                        customer_id TEXT NOT NULL,
                        opportunity_id TEXT NOT NULL,
                        run_id TEXT NOT NULL,
                        memory_type TEXT NOT NULL,
                        content JSONB NOT NULL,
                        source TEXT NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        UNIQUE (
                            customer_id,
                            opportunity_id,
                            run_id,
                            memory_type
                        )
                    )
                    """
                )
                await cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_sales_memories_customer
                    ON sales_memories(customer_id, created_at DESC)
                    """
                )

    async def save(
        self,
        *,
        customer_id: str,
        opportunity_id: str,
        run_id: str,
        memory_type: str,
        content: dict[str, Any],
        source: str,
    ) -> int:
        serialized = json.dumps(content, ensure_ascii=False)

        connection = await self._connect()
        async with connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """
                    INSERT INTO sales_memories (
                        customer_id,
                        opportunity_id,
                        run_id,
                        memory_type,
                        content,
                        source
                    ) VALUES (%s, %s, %s, %s, %s::jsonb, %s)
                    ON CONFLICT (
                        customer_id,
                        opportunity_id,
                        run_id,
                        memory_type
                    ) DO UPDATE SET
                        content = EXCLUDED.content,
                        source = EXCLUDED.source,
                        created_at = NOW()
                    RETURNING id
                    """,
                    (
                        customer_id,
                        opportunity_id,
                        run_id,
                        memory_type,
                        serialized,
                        source,
                    ),
                )
                row = await cursor.fetchone()

        if row is None:
            raise RuntimeError("长期记忆写入失败")
        return int(row["id"])

    async def search(
        self,
        customer_id: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        connection = await self._connect()
        async with connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """
                    SELECT id, opportunity_id, run_id, memory_type,
                           content, source, created_at
                    FROM sales_memories
                    WHERE customer_id = %s
                    ORDER BY created_at DESC, id DESC
                    LIMIT %s
                    """,
                    (customer_id, limit),
                )
                rows = await cursor.fetchall()

        return [
            {
                "id": row["id"],
                "opportunity_id": row["opportunity_id"],
                "run_id": row["run_id"],
                "memory_type": row["memory_type"],
                "content": (
                    json.loads(row["content"])
                    if isinstance(row["content"], str)
                    else row["content"]
                ),
                "source": row["source"],
                "created_at": row["created_at"].isoformat(),
            }
            for row in rows
        ]
