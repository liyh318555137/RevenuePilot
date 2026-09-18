import json
import os
import sys
import asyncio
from functools import lru_cache
from uuid import uuid4
from typing import TypedDict, Literal

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from tools import (
    get_crm_record,
    get_meeting_notes,
    get_product_catalog,
)
from context_manager import ContextBuilder
from harness import AgentHarness, build_thread_id
from memory import SalesMemory
from policy import HUMAN_APPROVAL_THRESHOLD
from knowledge_store import (
    EmbeddingClient,
    KnowledgeStore,
    knowledge_enabled,
)


MAX_DEBATE_ROUNDS = 2


load_dotenv()

if os.name == "nt":
    asyncio.set_event_loop_policy(
        asyncio.WindowsSelectorEventLoopPolicy()
    )

context_builder = ContextBuilder()


class SalesDecision(BaseModel):

    decision: Literal[
        "优先推进",
        "潜在机会",
        "建议终止",
    ]

    confidence: int = Field(
        ge=0,
        le=100,
    )

    reasons: list[str]

    opportunities: list[str]

    risks: list[str]

    next_actions: list[str]

    unknowns: list[str]


class SalesState(TypedDict):

    customer_id: str
    tenant_id: str
    opportunity_id: str
    run_id: str
    company: str

    crm_data: dict
    meeting_notes: str
    product_catalog: list
    retrieved_memories: list[dict]

    knowledge_query: str
    conversation_evidence: list[dict]
    case_evidence: list[dict]
    sop_evidence: list[dict]
    retrieval_warnings: list[str]

    missing_fields: list[str]
    data_quality_report: str

    account_report: str
    budget_report: str
    intent_report: str
    product_fit_report: str

    advocate_report: str
    skeptic_report: str
    debate_history: list[str]
    debate_round: int

    strategy_report: str
    final_decision: dict

    approval_status: str
    approval_comment: str
    memory_saved: bool


def log(level, message):

    print(
        f"[{level}] {message}"
    )


@lru_cache(maxsize=1)
def get_harness():

    api_key = os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        raise ValueError(
            "没有找到 DEEPSEEK_API_KEY，请检查 .env 文件"
        )

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )

    return AgentHarness(
        client=client,
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
        logger=log,
    )


def get_database_url():

    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            "没有找到 DATABASE_URL，请检查 .env 文件"
        )

    return database_url


@lru_cache(maxsize=1)
def get_memory_store():

    return SalesMemory(
        get_database_url()
    )


@lru_cache(maxsize=1)
def get_knowledge_store():
    return KnowledgeStore(
        get_database_url(),
        embedding_dimensions=int(
            os.getenv("EMBEDDING_DIMENSIONS", "1536")
        ),
    )


@lru_cache(maxsize=1)
def get_embedding_client():
    api_key = (
        os.getenv("EMBEDDING_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    if not api_key:
        raise ValueError(
            "知识库已启用，但没有找到 EMBEDDING_API_KEY 或 OPENAI_API_KEY"
        )
    return EmbeddingClient(
        api_key=api_key,
        model=os.getenv(
            "EMBEDDING_MODEL",
            "text-embedding-3-small",
        ),
        base_url=os.getenv("EMBEDDING_BASE_URL") or None,
    )


def load_opportunities():

    with open(
        "data/opportunities.json",
        "r",
        encoding="utf-8",
    ) as file:

        opportunities = json.load(file)

    return opportunities


def find_opportunity(
    opportunities,
    keyword,
):

    target = str(keyword).strip().lower()

    if not target:
        return None

    for customer in opportunities:

        for field in (
            "id",
            "customer_id",
            "opportunity_id",
            "company",
        ):

            value = customer.get(field)

            if value is None:
                continue

            if (
                str(value).strip().lower()
                == target
            ):
                return customer

    return None


def print_opportunity_list(
    opportunities
):

    print("\n可选客户：")

    for customer in opportunities:

        print(
            "  {}  {}  {}".format(
                customer.get(
                    "customer_id",
                    customer.get("id", ""),
                ),
                customer.get(
                    "opportunity_id",
                    "",
                ),
                customer.get("company", ""),
            )
        )

    print()


async def select_opportunity(
    opportunities,
    keyword=None,
):

    if keyword:

        customer = find_opportunity(
            opportunities,
            keyword,
        )

        if customer is None:

            print(
                f"未找到匹配「{keyword}」的客户。"
            )

            print_opportunity_list(
                opportunities
            )

            raise SystemExit(1)

        return customer

    print_opportunity_list(
        opportunities
    )

    while True:

        try:

            answer = (
                await asyncio.to_thread(
                    input,
                    "请输入客户 id 或公司名称"
                    "（直接回车退出）：",
                )
            ).strip()

        except EOFError:

            print("\n没有读到输入，已退出。")

            raise SystemExit(1)

        if not answer:

            raise SystemExit(0)

        customer = find_opportunity(
            opportunities,
            answer,
        )

        if customer is not None:
            return customer

        print(f"\n未找到「{answer}」。")

        similar = [
            item
            for item in opportunities
            if (
                answer.lower()
                in str(
                    item.get("company", "")
                ).lower()
            )
        ]

        if similar:

            print("你是不是想找：")

            for item in similar:

                print(
                    "  {}  {}".format(
                        item.get(
                            "customer_id",
                            item.get("id", ""),
                        ),
                        item.get("company", ""),
                    )
                )

        print()


def build_run_identity(
    opportunity_id,
    customer_id,
):
    """run_id 每次运行唯一；thread_id 按客户稳定，二者解耦。"""

    override = os.getenv("SALES_RUN_ID")

    return {
        "run_id":
            override or uuid4().hex,

        "thread_id":
            build_thread_id(
                opportunity_id,
                override or customer_id,
            ),
    }


async def call_llm(
    prompt,
    agent_name="sales_agent",
    max_retries=3,
):
    return await get_harness().run_text(
        prompt,
        agent_name=agent_name,
        max_retries=max_retries,
    )


async def call_llm_json(
    prompt,
    agent_name="sales_manager",
    max_retries=3,
):
    return await get_harness().run_json(
        prompt,
        agent_name=agent_name,
        max_retries=max_retries,
    )


async def load_sales_data(
    state: SalesState
):

    log(
        "NODE",
        "Load Sales Data 开始"
    )

    customer_id = state[
        "customer_id"
    ]

    (
        crm_data,
        meeting_notes,
        product_catalog,
    ) = await asyncio.gather(
        asyncio.to_thread(
            get_crm_record,
            customer_id,
        ),
        asyncio.to_thread(
            get_meeting_notes,
            customer_id,
        ),
        asyncio.to_thread(
            get_product_catalog,
        ),
    )

    log(
        "NODE",
        "Load Sales Data 完成"
    )

    return {
        "crm_data": crm_data or {},
        "meeting_notes": (
            meeting_notes or ""
        ),
        "product_catalog": (
            product_catalog or []
        ),
    }


async def retrieve_customer_memory(
    state: SalesState
):

    log(
        "NODE",
        "Retrieve Customer Memory 开始"
    )

    memories = await get_memory_store().search(
        customer_id=state["customer_id"],
        limit=5,
    )

    log(
        "NODE",
        f"读取到 {len(memories)} 条长期记忆"
    )

    return {
        "retrieved_memories": memories
    }


def check_data_quality(
    state: SalesState
):

    log(
        "NODE",
        "Check Data Quality 开始"
    )

    missing_fields = []

    if not state["crm_data"]:

        missing_fields.append(
            "crm_data"
        )

    if not state[
        "meeting_notes"
    ]:

        missing_fields.append(
            "meeting_notes"
        )

    if not state[
        "product_catalog"
    ]:

        missing_fields.append(
            "product_catalog"
        )

    log(
        "NODE",
        f"缺失字段：{missing_fields}"
    )

    return {
        "missing_fields":
            missing_fields
    }


def route_after_data_check(
    state: SalesState
):

    log(
        "ROUTER",
        "判断数据完整性"
    )

    if state["missing_fields"]:

        log(
            "ROUTER",
            "→ missing_data_handler"
        )

        return (
            "missing_data_handler"
        )

    log(
        "ROUTER",
        "→ start_analysis"
    )

    return "start_analysis"


def missing_data_handler(
    state: SalesState
):

    log(
        "NODE",
        "Missing Data Handler"
    )

    missing_fields = state[
        "missing_fields"
    ]

    report = (
        "当前销售机会存在数据缺失。"
        f"缺失字段：{missing_fields}。"
        "后续所有分析必须明确说明"
        "这些信息未知，不能自行假设。"
    )

    return {
        "data_quality_report":
            report
    }


def build_knowledge_query(state: SalesState) -> str:
    """Create one grounded query shared by the three knowledge corpora."""

    crm = state.get("crm_data") or {}
    return "\n".join(
        [
            f"公司：{crm.get('company', state.get('company', '未知'))}",
            f"行业：{crm.get('industry', '未知')}",
            f"销售阶段：{crm.get('stage', '未知')}",
            f"预算：{crm.get('budget', '未知')}",
            f"交易金额：{crm.get('deal_amount', '未知')}",
            f"竞争对手：{crm.get('competitor', '未知')}",
            f"会议记录：{state.get('meeting_notes') or '未知'}",
        ]
    )


async def retrieve_sales_knowledge(
    state: SalesState,
):
    """Retrieve scoped RAG evidence once before the analysis fan-out."""

    log("NODE", "Retrieve Sales Knowledge 开始")
    empty = {
        "knowledge_query": "",
        "conversation_evidence": [],
        "case_evidence": [],
        "sop_evidence": [],
        "retrieval_warnings": [],
    }

    if not knowledge_enabled():
        log("RAG", "知识库未启用，跳过检索")
        return empty

    query = build_knowledge_query(state)
    try:
        embedding = await get_embedding_client().embed(query)
        store = get_knowledge_store()
        conversation, cases, sop = await asyncio.gather(
            store.search(
                tenant_id=state.get("tenant_id", "default"),
                corpus_type="customer_conversation",
                customer_id=state["customer_id"],
                strict_customer=True,
                query=query,
                embedding=embedding,
                limit=5,
            ),
            store.search(
                tenant_id=state.get("tenant_id", "default"),
                corpus_type="success_case",
                customer_id=state["customer_id"],
                query=query,
                embedding=embedding,
                limit=5,
            ),
            store.search(
                tenant_id=state.get("tenant_id", "default"),
                corpus_type="sales_sop",
                customer_id=state["customer_id"],
                query=query,
                embedding=embedding,
                limit=5,
            ),
        )
        log(
            "RAG",
            f"检索完成：对话 {len(conversation)}，案例 {len(cases)}，SOP {len(sop)}",
        )
        return {
            "knowledge_query": query,
            "conversation_evidence": conversation,
            "case_evidence": cases,
            "sop_evidence": sop,
            "retrieval_warnings": [],
        }
    except Exception as exc:
        warning = f"知识库检索失败，已降级为空证据：{exc}"
        log("RAG", warning)
        return {
            **empty,
            "knowledge_query": query,
            "retrieval_warnings": [warning],
        }


def start_analysis(
    _state: SalesState
):

    log(
        "NODE",
        "并发启动 Account 分析链与 Product Fit 分析"
    )

    return {}


async def account_analyst(
    state: SalesState
):

    log(
        "NODE",
        "Account Analyst 开始"
    )

    context = context_builder.build(
        "account_analyst",
        state,
    )

    customer = context["customer"]

    prompt = f"""
你是一名 Account Analyst。

客户基础信息：

公司：
{customer["company"]}

交易金额：
{customer["deal_amount"]}

销售阶段：
{customer["stage"]}

预算：
{customer["budget"]}

竞争情况：
{customer["competitor"]}

决策人：
{customer["decision_maker"]}

会议记录：

{context["meeting_notes"]}

数据质量报告：

{context["data_quality_report"]}

客户长期记忆（仅作为历史参考）：

{context["customer_memories"]}

相关客户对话证据（仅作为事实参考）：

{context["conversation_evidence"]}

请分析：

1. 客户当前状态
2. 当前正面信号
3. 当前负面信号
4. 缺少哪些关键信息

不要做最终销售决策。
不要虚构任何信息。
"""

    result = await call_llm(
        prompt,
        agent_name="account_analyst",
    )

    log(
        "NODE",
        "Account Analyst 完成"
    )

    return {
        "account_report": result
    }


async def budget_analyst(
    state: SalesState
):

    log(
        "NODE",
        "Budget Analyst 开始"
    )

    context = context_builder.build(
        "budget_analyst",
        state,
    )

    prompt = f"""
你是一名 Budget Analyst。

交易金额：

{context["deal_amount"]}

预算情况：

{context["budget"]}

Account Analyst Report：

{context["account_report"]}

历史决策记忆：

{context["customer_memories"]}

相关销售 SOP 证据：

{context["sop_evidence"]}

请分析：

1. 预算是否匹配交易金额
2. 预算信息是否明确
3. 当前最大的预算风险
4. 还需要确认哪些预算问题

不要做最终销售决策。
不要虚构财务信息。
"""

    result = await call_llm(
        prompt,
        agent_name="budget_analyst",
    )

    log(
        "NODE",
        "Budget Analyst 完成"
    )

    return {
        "budget_report": result
    }


async def intent_analyst(
    state: SalesState
):

    log(
        "NODE",
        "Intent Analyst 开始"
    )

    context = context_builder.build(
        "intent_analyst",
        state,
    )

    prompt = f"""
你是一名 Intent Analyst。

会议记录：

{context["meeting_notes"]}

====================
Account Report
====================

{context["account_report"]}

====================
Budget Report
====================

{context["budget_report"]}

历史决策记忆：

{context["customer_memories"]}

相关客户对话证据：

{context["conversation_evidence"]}

相关销售 SOP 证据：

{context["sop_evidence"]}

请分析：

1. 客户购买意向：高 / 中 / 低
2. 支持这一判断的事实
3. 项目成熟度
4. 仍然需要验证哪些信号

不要做最终销售决策。
不要虚构信息。
"""

    result = await call_llm(
        prompt,
        agent_name="intent_analyst",
    )

    log(
        "NODE",
        "Intent Analyst 完成"
    )

    return {
        "intent_report": result
    }


async def product_fit_analyst(
    state: SalesState
):

    log(
        "NODE",
        "Product Fit Analyst 开始"
    )

    context = context_builder.build(
        "product_fit_analyst",
        state,
    )

    prompt = f"""
你是一名 Product Fit Analyst。

客户：

{context["company"]}

交易金额：

{context["deal_amount"]}

预算：

{context["budget"]}

客户会议：

{context["meeting_notes"]}

产品目录：

{context["product_catalog"]}

客户历史决策记忆：

{context["customer_memories"]}

相关成功案例：

{context["case_evidence"]}

请分析：

1. 最匹配的产品
2. 匹配原因
3. 预算是否匹配
4. 产品与需求之间可能存在的差距
5. 仍需要确认什么

不要做最终“优先推进”/
“潜在机会”/“建议终止”决策。

不要虚构产品目录中不存在的能力。
"""

    result = await call_llm(
        prompt,
        agent_name="product_fit_analyst",
    )

    log(
        "NODE",
        "Product Fit Analyst 完成"
    )

    return {
        "product_fit_report":
            result
    }


async def deal_advocate(
    state: SalesState
):

    context = context_builder.build(
        "deal_advocate",
        state,
    )

    round_number = (
        context["debate_round"]
        + 1
    )

    log(
        "NODE",
        (
            "Deal Advocate "
            f"Round {round_number}"
        )
    )

    customer = context["customer"]

    if context["skeptic_report"]:

        skeptic_context = f"""
上一轮 Skeptic 的观点：

{context["skeptic_report"]}

请针对其中有依据的风险进行回应。
"""

    else:

        skeptic_context = """
这是第一轮。

Skeptic 还没有提出观点。

不要假装正在反驳一个不存在的观点。
"""

    prompt = f"""
你是一名 Deal Advocate。

你的职责：

基于现有事实，
寻找这个销售机会值得继续投入资源的证据。

你不能无条件乐观。

客户：

{customer["company"]}

交易金额：

{customer["deal_amount"]}

预算：

{customer["budget"]}

竞争情况：

{customer["competitor"]}

====================
Account Report
====================

{context["account_report"]}

====================
Budget Report
====================

{context["budget_report"]}

====================
Intent Report
====================

{context["intent_report"]}

====================
Product Fit Report
====================

{context["product_fit_report"]}

相关成功案例（仅作为可比证据）：

{context["case_evidence"]}

{skeptic_context}

请分析：

1. 为什么这个机会仍值得推进
2. 最强的三个正面证据
3. 哪些风险可以通过行动降低
4. 下一步最值得投入资源的动作
5. 哪些观点仍然证据不足

不要虚构事实。

不要做最终
“优先推进”/“潜在机会”/“建议终止”
决策。
"""

    result = await call_llm(
        prompt,
        agent_name="deal_advocate",
    )

    new_history = (
        state["debate_history"]
        + [
            (
                f"Round {round_number}"
                f" - Advocate:\n{result}"
            )
        ]
    )

    return {
        "advocate_report": result,
        "debate_history":
            new_history,
    }


async def deal_skeptic(
    state: SalesState
):

    context = context_builder.build(
        "deal_skeptic",
        state,
    )

    round_number = (
        context["debate_round"]
        + 1
    )

    log(
        "NODE",
        (
            "Deal Skeptic "
            f"Round {round_number}"
        )
    )

    customer = context["customer"]

    prompt = f"""
你是一名 Deal Skeptic。

你的职责不是为了反对而反对。

你的职责是发现：

- 被销售人员忽视的风险
- 证据不足的判断
- 不值得继续投入资源的因素

客户：

{customer["company"]}

交易金额：

{customer["deal_amount"]}

预算：

{customer["budget"]}

竞争情况：

{customer["competitor"]}

====================
Advocate 当前观点
====================

{context["advocate_report"]}

====================
Account Report
====================

{context["account_report"]}

====================
Budget Report
====================

{context["budget_report"]}

====================
Intent Report
====================

{context["intent_report"]}

====================
Product Fit Report
====================

{context["product_fit_report"]}

相关成功案例（仅作为反事实校验参考）：

{context["case_evidence"]}

请分析：

1. Advocate 哪些观点证据充分
2. 哪些观点证据不足
3. 三个最危险的成交风险
4. 哪些假设必须验证
5. 什么情况下应该减少销售投入

不要为了反驳而虚构事实。

不要做最终
“优先推进”/“潜在机会”/“建议终止”
决策。
"""

    result = await call_llm(
        prompt,
        agent_name="deal_skeptic",
    )

    new_history = (
        state["debate_history"]
        + [
            (
                f"Round {round_number}"
                f" - Skeptic:\n{result}"
            )
        ]
    )

    return {
        "skeptic_report": result,

        "debate_history":
            new_history,

        "debate_round":
            round_number,
    }


def route_after_debate(
    state: SalesState
):

    current_round = state[
        "debate_round"
    ]

    log(
        "ROUTER",
        f"当前 Debate Round = {current_round}"
    )

    if (
        current_round
        < MAX_DEBATE_ROUNDS
    ):

        log(
            "ROUTER",
            "继续 → deal_advocate"
        )

        return "deal_advocate"

    log(
        "ROUTER",
        "结束 → strategy_manager"
    )

    return "strategy_manager"


async def strategy_manager(
    state: SalesState
):

    log(
        "NODE",
        "Strategy Manager 开始"
    )

    context = context_builder.build(
        "strategy_manager",
        state,
    )

    debate_text = (
        "\n\n"
        .join(
            context["debate_history"]
        )
    )

    prompt = f"""
你是一名 Sales Strategy Manager。

你的职责不是投票决定 Advocate
还是 Skeptic 胜利。

你需要根据证据形成销售策略。

客户：

{context["company"]}

====================
完整 Debate History
====================

{debate_text}

====================
Product Fit Report
====================

{context["product_fit_report"]}

相关成功案例：

{context["case_evidence"]}

相关销售 SOP：

{context["sop_evidence"]}

请输出：

1. Advocate 最有价值的证据
2. Skeptic 最重要的风险
3. 双方哪些判断证据不足
4. 当前最合理的销售策略
5. 最重要的三个验证动作
6. 什么条件变化后需要重新调整策略

必须依据事实。

不要简单投票。

不要虚构信息。
"""

    result = await call_llm(
        prompt,
        agent_name="strategy_manager",
    )

    log(
        "NODE",
        "Strategy Manager 完成"
    )

    return {
        "strategy_report":
            result
    }


async def sales_manager(
    state: SalesState
):

    log(
        "NODE",
        "Sales Manager 开始"
    )

    context = context_builder.build(
        "sales_manager",
        state,
    )

    customer = context["customer"]

    prompt = f"""
你是一名 Sales Manager。

你的任务是做最终销售决策。

Strategy Manager 的结果只是输入之一，
不是不可质疑的最终答案。

你必须综合：

- 原始客户事实
- Data Quality
- Analyst Reports
- Debate
- Strategy Manager

====================
Data Quality
====================

{context["data_quality_report"]}

RAG Retrieval Warnings
====================

{context["retrieval_warnings"]}

====================
Account Analyst
====================

{context["account_report"]}

====================
Budget Analyst
====================

{context["budget_report"]}

====================
Intent Analyst
====================

{context["intent_report"]}

====================
Product Fit Analyst
====================

{context["product_fit_report"]}

====================
Strategy Manager
====================

{context["strategy_report"]}

====================
Customer Memory
====================

{context["customer_memories"]}

Conversation Evidence
====================

{context["conversation_evidence"]}

Success Case Evidence
====================

{context["case_evidence"]}

Sales SOP Evidence
====================

{context["sop_evidence"]}

客户：

{customer["company"]}

交易金额：

{customer["deal_amount"]}

预算：

{customer["budget"]}

竞争情况：

{customer["competitor"]}

决策人：

{customer["decision_maker"]}

请严格输出 JSON。

格式：

{{
    "decision":
        "优先推进 或 潜在机会 或 建议终止",

    "confidence":
        0到100之间的整数,

    "reasons": [
        "理由1",
        "理由2"
    ],

    "opportunities": [
        "机会1",
        "机会2",
        "机会3"
    ],

    "risks": [
        "风险1",
        "风险2",
        "风险3"
    ],

    "next_actions": [
        "行动1",
        "行动2",
        "行动3"
    ],

    "unknowns": [
        "未知信息"
    ]
}}

不要输出 JSON 以外的文字。

不要虚构信息。
"""

    raw_result = (
        await call_llm_json(
            prompt,
            agent_name="sales_manager",
        )
    )

    validated_result = (
        SalesDecision
        .model_validate(
            raw_result
        )
    )

    result = (
        validated_result
        .model_dump()
    )

    log(
        "NODE",
        "Sales Manager 完成"
    )

    return {
        "final_decision":
            result
    }


def route_after_sales_manager(
    state: SalesState
):

    log(
        "ROUTER",
        "检查是否需要人工审批"
    )

    decision = (
        state[
            "final_decision"
        ]
        .get(
            "decision"
        )
    )

    crm = (
        state["crm_data"]
        or {}
    )

    deal_amount = (
        crm.get(
            "deal_amount",
            0,
        )
    )

    if (
        decision == "优先推进"
        and deal_amount >= HUMAN_APPROVAL_THRESHOLD
    ):

        log(
            "ROUTER",
            "高风险决策 → human_review"
        )

        return "human_review"

    log(
        "ROUTER",
        "无需人工审批 → END"
    )

    return "end"


def human_review(
    state: SalesState
):

    log(
        "HUMAN",
        "等待人工审批"
    )

    crm = (
        state["crm_data"]
        or {}
    )

    decision = state[
        "final_decision"
    ]

    human_result = interrupt(
        {
            "message":
                "该销售机会需要人工审批",

            "company":
                crm.get(
                    "company",
                    state["company"],
                ),

            "deal_amount":
                crm.get(
                    "deal_amount",
                    0,
                ),

            "decision":
                decision.get(
                    "decision"
                ),

            "confidence":
                decision.get(
                    "confidence"
                ),

            "risks":
                decision.get(
                    "risks",
                    [],
                ),
        }
    )

    approved = (
        human_result
        .get(
            "approved",
            False,
        )
    )

    comment = (
        human_result
        .get(
            "comment",
            "",
        )
    )

    if approved:

        status = "APPROVED"

    else:

        status = "REJECTED"

    log(
        "HUMAN",
        f"人工审批结果：{status}"
    )

    return {
        "approval_status":
            status,

        "approval_comment":
            comment,
    }


async def extract_and_save_memory(
    state: SalesState
):

    log(
        "NODE",
        "Extract And Save Memory 开始"
    )

    decision = state.get(
        "final_decision",
        {},
    )

    if not decision:
        log(
            "MEMORY",
            "没有最终决策，跳过长期记忆写入"
        )
        return {
            "memory_saved": False
        }

    memory_id = await get_memory_store().save(
        customer_id=state["customer_id"],
        opportunity_id=state["opportunity_id"],
        run_id=state["run_id"],
        memory_type="sales_decision",
        content={
            "company": state["company"],
            "decision": decision.get("decision"),
            "confidence": decision.get("confidence"),
            "reasons": decision.get("reasons", []),
            "risks": decision.get("risks", []),
            "next_actions": decision.get("next_actions", []),
            "unknowns": decision.get("unknowns", []),
            "approval_status": state.get("approval_status", ""),
            "approval_comment": state.get("approval_comment", ""),
        },
        source="sales_workflow",
    )

    log(
        "MEMORY",
        f"长期记忆已保存，ID={memory_id}"
    )

    return {
        "memory_saved": True
    }


def build_graph(
    checkpointer
):

    workflow = StateGraph(
        SalesState
    )

    workflow.add_node(
        "load_sales_data",
        load_sales_data,
    )

    workflow.add_node(
        "retrieve_customer_memory",
        retrieve_customer_memory,
    )

    workflow.add_node(
        "check_data_quality",
        check_data_quality,
    )

    workflow.add_node(
        "missing_data_handler",
        missing_data_handler,
    )

    workflow.add_node(
        "retrieve_sales_knowledge",
        retrieve_sales_knowledge,
    )

    workflow.add_node(
        "start_analysis",
        start_analysis,
    )

    workflow.add_node(
        "account_analyst",
        account_analyst,
    )

    workflow.add_node(
        "budget_analyst",
        budget_analyst,
    )

    workflow.add_node(
        "intent_analyst",
        intent_analyst,
    )

    workflow.add_node(
        "product_fit_analyst",
        product_fit_analyst,
    )

    workflow.add_node(
        "deal_advocate",
        deal_advocate,
    )

    workflow.add_node(
        "deal_skeptic",
        deal_skeptic,
    )

    workflow.add_node(
        "strategy_manager",
        strategy_manager,
    )

    workflow.add_node(
        "sales_manager",
        sales_manager,
    )

    workflow.add_node(
        "human_review",
        human_review,
    )

    workflow.add_node(
        "extract_and_save_memory",
        extract_and_save_memory,
    )

    workflow.add_edge(
        START,
        "load_sales_data",
    )

    workflow.add_edge(
        "load_sales_data",
        "retrieve_customer_memory",
    )

    workflow.add_edge(
        "retrieve_customer_memory",
        "check_data_quality",
    )

    workflow.add_conditional_edges(
        "check_data_quality",
        route_after_data_check,
        {
            "missing_data_handler":
                "missing_data_handler",

            "start_analysis":
                "retrieve_sales_knowledge",
        },
    )

    workflow.add_edge(
        "missing_data_handler",
        "retrieve_sales_knowledge",
    )

    workflow.add_edge(
        "retrieve_sales_knowledge",
        "start_analysis",
    )

    workflow.add_edge(
        "start_analysis",
        "account_analyst",
    )

    workflow.add_edge(
        "start_analysis",
        "product_fit_analyst",
    )

    workflow.add_edge(
        "account_analyst",
        "budget_analyst",
    )

    workflow.add_edge(
        "budget_analyst",
        "intent_analyst",
    )

    workflow.add_edge(
        [
            "intent_analyst",
            "product_fit_analyst",
        ],
        "deal_advocate",
    )

    workflow.add_edge(
        "deal_advocate",
        "deal_skeptic",
    )

    workflow.add_conditional_edges(
        "deal_skeptic",
        route_after_debate,
        {
            "deal_advocate":
                "deal_advocate",

            "strategy_manager":
                "strategy_manager",
        },
    )

    workflow.add_edge(
        "strategy_manager",
        "sales_manager",
    )

    workflow.add_conditional_edges(
        "sales_manager",
        route_after_sales_manager,
        {
            "human_review":
                "human_review",

            "end":
                "extract_and_save_memory",
        },
    )

    workflow.add_edge(
        "human_review",
        "extract_and_save_memory",
    )

    workflow.add_edge(
        "extract_and_save_memory",
        END,
    )

    graph = workflow.compile(
        checkpointer=checkpointer
    )

    return graph


def create_initial_state(
    customer,
    run_id=None,
):

    customer_id = customer.get(
        "customer_id",
        customer["id"],
    )
    opportunity_id = customer.get(
        "opportunity_id",
        customer["id"],
    )
    current_run_id = run_id or uuid4().hex

    return {

        "customer_id":
            customer_id,

        "tenant_id":
            customer.get(
                "tenant_id",
                os.getenv("SALES_TENANT_ID", "default"),
            ),

        "opportunity_id":
            opportunity_id,

        "run_id":
            current_run_id,

        "company":
            customer["company"],

        "crm_data": {},

        "meeting_notes": "",

        "product_catalog": [],

        "retrieved_memories": [],

        "knowledge_query": "",

        "conversation_evidence": [],

        "case_evidence": [],

        "sop_evidence": [],

        "retrieval_warnings": [],

        "missing_fields": [],

        "data_quality_report": "",

        "account_report": "",

        "budget_report": "",

        "intent_report": "",

        "product_fit_report": "",

        "advocate_report": "",

        "skeptic_report": "",

        "debate_history": [],

        "debate_round": 0,

        "strategy_report": "",

        "final_decision": {},

        "approval_status": "",

        "approval_comment": "",

        "memory_saved": False,
    }


def print_final_result(
    final_state
):

    print("\n")
    print("=" * 60)
    print("FINAL RESULT")
    print("=" * 60)

    decision = (
        final_state
        .get(
            "final_decision",
            {},
        )
    )

    print(
        "Decision:",
        decision.get(
            "decision"
        )
    )

    print(
        "Confidence:",
        decision.get(
            "confidence"
        )
    )

    print(
        "Approval Status:",
        final_state.get(
            "approval_status",
            "",
        )
    )

    print(
        "Approval Comment:",
        final_state.get(
            "approval_comment",
            "",
        )
    )

    print("\nRisks:")

    for risk in decision.get(
        "risks",
        [],
    ):

        print(
            "-",
            risk
        )

    print("\nNext Actions:")

    for action in decision.get(
        "next_actions",
        [],
    ):

        print(
            "-",
            action
        )


async def main():

    opportunities = (
        load_opportunities()
    )

    customer = await select_opportunity(
        opportunities,
        sys.argv[1]
        if len(sys.argv) > 1
        else None,
    )

    opportunity_id = customer.get(
        "opportunity_id",
        customer["id"],
    )

    customer_id = customer.get(
        "customer_id",
        customer["id"],
    )

    identity = build_run_identity(
        opportunity_id,
        customer_id,
    )

    run_id = identity["run_id"]

    thread_id = identity["thread_id"]

    config = {
        "configurable": {
            "thread_id":
                thread_id
        }
    }

    print("=" * 60)

    print(
        "Customer:",
        customer["company"]
    )

    print(
        "Thread ID:",
        thread_id
    )

    print(
        "Run ID:",
        run_id
    )

    print("=" * 60)

    async with (
        AsyncPostgresSaver
        .from_conn_string(
            get_database_url()
        )
    ) as checkpointer:

        await checkpointer.setup()
        await get_memory_store().setup()
        if knowledge_enabled():
            await get_knowledge_store().setup()

        graph = build_graph(
            checkpointer
        )

        pending = await graph.aget_state(
            config
        )

        if pending.next:

            log(
                "RESUME",
                "检测到未完成的运行，从检查点继续，不重跑已完成节点"
            )

            graph_input = None

        else:

            graph_input = (
                create_initial_state(
                    customer,
                    run_id=run_id,
                )
            )

        result = await graph.ainvoke(
            graph_input,
            config=config,
        )

        snapshot = await graph.aget_state(
            config
        )

        print(
            "\n=== CHECKPOINT ==="
        )

        print(
            "Next:",
            snapshot.next
        )

        print(
            "Debate Round:",
            snapshot.values.get(
                "debate_round"
            )
        )

        interrupts = result.get(
            "__interrupt__",
            []
        )

        if interrupts:

            print("\n")
            print("=" * 60)
            print("需要人工审批")
            print("=" * 60)

            print(
                interrupts
            )

            print(
                "\n你现在可以直接 Ctrl + C，"
                f"然后重新运行「app.py {customer_id}」"
                "，会从这个检查点继续。"
            )

            print()

            answer = await asyncio.to_thread(
                input,
                "是否批准？yes/no：",
            )

            comment = await asyncio.to_thread(
                input,
                "审批意见：",
            )

            approved = (
                answer
                .strip()
                .lower()
                == "yes"
            )

            result = await graph.ainvoke(
                Command(
                    resume={
                        "approved":
                            approved,

                        "comment":
                            comment,
                    }
                ),
                config=config,
            )

        final_snapshot = (
            await graph.aget_state(
                config
            )
        )

        final_state = (
            final_snapshot.values
        )

        print_final_result(
            final_state
        )

        history = [
            item
            async for item in (
                graph.aget_state_history(
                    config
                )
            )
        ]

        print(
            "\nCheckpoint 数量：",
            len(history)
        )


if __name__ == "__main__":
    asyncio.run(main())
