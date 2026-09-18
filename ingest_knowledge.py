"""Ingest local sales documents into the PostgreSQL/pgvector knowledge base.

Example:
    python ingest_knowledge.py ./knowledge --corpus-type sales_sop

The command is intentionally separate from ``app.py``.  It can be run when a
document is added or changed, while customer decision runs only perform fast
retrieval.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from dotenv import load_dotenv

from knowledge_store import EmbeddingClient, KnowledgeStore


SUPPORTED_SUFFIXES = {".txt", ".md", ".json"}


def read_document(path: Path) -> str:
    if path.suffix.lower() == ".json":
        return json.dumps(
            json.loads(path.read_text(encoding="utf-8")),
            ensure_ascii=False,
            indent=2,
        )
    return path.read_text(encoding="utf-8")


CORPUS_DIR_NAMES = {"customer_conversation", "success_case", "sales_sop"}


def _known_customer_ids() -> set[str]:
    """尽量从 data/ 读出已知客户 id，读不到就返回空集合。

    只用于交叉污染的启发式检查；读不到不应该让摄取失败。
    """

    ids: set[str] = set()

    for name in ("opportunities.json", "crm.json"):
        try:
            records = json.loads(
                (Path("data") / name).read_text(encoding="utf-8")
            )
        except Exception:
            continue

        if not isinstance(records, list):
            continue

        for record in records:
            if not isinstance(record, dict):
                continue
            for key in ("customer_id", "id"):
                if record.get(key):
                    ids.add(str(record[key]))

    return ids


def check_customer_scope(root: Path, customer_id: str) -> str | None:
    """检查 `--customer-id` 有没有被用在过宽的根目录上。

    把 `./knowledge` 这样的整个知识库绑定到单个 customer_id，会把其他客户的
    对话一并标成本客户的记录。这种污染在检索侧完全看不出来：`customer_id`
    过滤会老老实实地返回这些片段，`reviewer` 的跨客户检查也就永远不会触发。
    返回问题描述，没有问题返回 None。
    """

    others = _known_customer_ids() - {customer_id}

    for path in sorted(root.rglob("*")):
        if not path.is_dir():
            continue

        if path.name in CORPUS_DIR_NAMES:
            return (
                f"根目录下出现了语料目录 {path.name}/，"
                f"说明 {root} 是整个知识库，而不是客户 {customer_id} 自己的目录"
            )

        if path.name in others:
            return (
                f"根目录下出现了其他客户目录 {path.name}/，"
                f"把这些文件标成 {customer_id} 会造成跨客户污染"
            )

    return None


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 150) -> list[str]:
    """Split on paragraphs first, keeping chunks bounded for retrieval."""

    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = current[-overlap:] + "\n\n" + paragraph
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def enforce_customer_scope(
    args: argparse.Namespace,
    root: Path | None = None,
) -> None:
    """摄取前挡住过宽的 `--customer-id`。"""

    if not args.customer_id or getattr(args, "allow_broad_root", False):
        return

    if root is None:
        root = Path(args.path).resolve()

    problem = check_customer_scope(root, args.customer_id)
    if not problem:
        return

    raise SystemExit(
        f"{problem}。\n"
        f"请改为只为该客户自己的目录摄取：\n"
        f"  python ingest_knowledge.py "
        f"./knowledge/customer_conversation/{args.customer_id} "
        f"--corpus-type customer_conversation "
        f"--customer-id {args.customer_id}\n"
        f"确认目录无误时，可加 --allow-broad-root 跳过此检查。"
    )


async def ingest(args: argparse.Namespace) -> int:
    load_dotenv()

    # 范围检查放在最前面：它只看本地目录，不需要网络和数据库。放到后面的话，
    # 凭证或连接先失败就会把这个问题盖住，而它一旦漏过去是数据污染。
    root = Path(args.path).resolve()
    enforce_customer_scope(args, root)

    api_key = os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("请在 .env 中配置 EMBEDDING_API_KEY 或 OPENAI_API_KEY")

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("请在 .env 中配置 DATABASE_URL")

    dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))
    store = KnowledgeStore(database_url, embedding_dimensions=dimensions)
    await store.setup()
    embedder = EmbeddingClient(
        api_key=api_key,
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        base_url=os.getenv("EMBEDDING_BASE_URL") or None,
    )

    paths = [
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    if not paths:
        print(f"未找到支持的文件：{root}")
        return 0

    metadata = json.loads(args.metadata) if args.metadata else {}
    chunks: list[dict] = []
    for path in paths:
        content = read_document(path).strip()
        if not content:
            continue
        relative_path = str(path.relative_to(root)).replace("\\", "/")
        # 每个客户通常单独摄取自己的目录，若不以 customer_id 作为前缀，
        # 所有客户都会得到 conversation.md 这样的相同 document_id，
        # UPSERT 键 (tenant_id, corpus_type, document_id, chunk_index) 会命中
        # ON CONFLICT，导致后摄取的客户覆盖前一个客户的片段。
        document_id = (
            f"{args.customer_id}/{relative_path}"
            if args.customer_id
            else relative_path
        )
        checksum = hashlib.sha256(content.encode("utf-8")).hexdigest()
        texts = chunk_text(content, max_chars=args.max_chars)
        embeddings = await asyncio.gather(*(embedder.embed(text) for text in texts))
        for index, (text, embedding) in enumerate(zip(texts, embeddings)):
            chunks.append(
                {
                    "id": str(
                        uuid5(
                            NAMESPACE_URL,
                            f"{args.tenant_id}:{args.corpus_type}:{document_id}:{index}",
                        )
                    ),
                    "tenant_id": args.tenant_id,
                    "corpus_type": args.corpus_type,
                    "customer_id": args.customer_id,
                    "document_id": document_id,
                    "chunk_index": index,
                    "content": text,
                    "embedding": embedding,
                    "metadata": {**metadata, "filename": path.name},
                    "source_uri": str(path),
                    "document_version": checksum[:16],
                    "checksum": checksum,
                }
            )

    written = await store.upsert_chunks(chunks)
    print(f"已写入 {written} 个知识片段，来源文件 {len(paths)} 个")
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="摄取销售知识库文件")
    parser.add_argument("path", help="包含 .txt/.md/.json 文件的目录")
    parser.add_argument(
        "--corpus-type",
        required=True,
        choices=("customer_conversation", "success_case", "sales_sop"),
    )
    parser.add_argument("--tenant-id", default=os.getenv("SALES_TENANT_ID", "default"))
    parser.add_argument("--customer-id")
    parser.add_argument(
        "--allow-broad-root",
        action="store_true",
        help="允许把包含其他客户目录的根目录整体绑定到 --customer-id",
    )
    parser.add_argument("--metadata", help="额外元数据 JSON，例如 '{\"industry\":\"制造\"}'")
    parser.add_argument("--max-chars", type=int, default=1200)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(ingest(parse_args()))
