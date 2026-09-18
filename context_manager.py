from typing import Any


class ContextBuilder:
    """为每个 Agent 节点选择最小必要上下文。"""

    def build(self, agent_name: str, state: dict[str, Any]) -> dict[str, Any]:
        builder = getattr(self, f"_for_{agent_name}", None)
        if builder is None:
            raise ValueError(f"未知的 Agent：{agent_name}")
        return builder(state)

    @staticmethod
    def _customer_facts(state: dict[str, Any]) -> dict[str, Any]:
        crm = state.get("crm_data") or {}
        return {
            "company": crm.get("company", state.get("company", "未知")),
            "deal_amount": crm.get("deal_amount", "未知"),
            "stage": crm.get("stage", "未知"),
            "budget": crm.get("budget", "未知"),
            "competitor": crm.get("competitor", "未知"),
            "decision_maker": crm.get("decision_maker", "未知"),
        }

    @staticmethod
    def _memories(state: dict[str, Any]) -> list[dict[str, Any]]:
        return list(state.get("retrieved_memories") or [])

    @staticmethod
    def _evidence(state: dict[str, Any], field: str) -> list[dict[str, Any]]:
        return list(state.get(field) or [])

    def _for_account_analyst(self, state: dict[str, Any]) -> dict[str, Any]:
        return {
            "customer": self._customer_facts(state),
            "meeting_notes": state.get("meeting_notes") or "未知",
            "data_quality_report": state.get("data_quality_report", ""),
            "customer_memories": self._memories(state),
            "conversation_evidence": self._evidence(
                state, "conversation_evidence"
            ),
        }

    def _for_budget_analyst(self, state: dict[str, Any]) -> dict[str, Any]:
        facts = self._customer_facts(state)
        return {
            "deal_amount": facts["deal_amount"],
            "budget": facts["budget"],
            "account_report": state.get("account_report", ""),
            "customer_memories": self._memories(state),
            "sop_evidence": self._evidence(state, "sop_evidence"),
        }

    def _for_intent_analyst(self, state: dict[str, Any]) -> dict[str, Any]:
        return {
            "meeting_notes": state.get("meeting_notes") or "未知",
            "account_report": state.get("account_report", ""),
            "budget_report": state.get("budget_report", ""),
            "customer_memories": self._memories(state),
            "conversation_evidence": self._evidence(
                state, "conversation_evidence"
            ),
            "sop_evidence": self._evidence(state, "sop_evidence"),
        }

    def _for_product_fit_analyst(self, state: dict[str, Any]) -> dict[str, Any]:
        facts = self._customer_facts(state)
        return {
            "company": facts["company"],
            "deal_amount": facts["deal_amount"],
            "budget": facts["budget"],
            "meeting_notes": state.get("meeting_notes") or "未知",
            "product_catalog": state.get("product_catalog") or [],
            "customer_memories": self._memories(state),
            "case_evidence": self._evidence(state, "case_evidence"),
        }

    def _for_deal_advocate(self, state: dict[str, Any]) -> dict[str, Any]:
        facts = self._customer_facts(state)
        return {
            "customer": facts,
            "account_report": state.get("account_report", ""),
            "budget_report": state.get("budget_report", ""),
            "intent_report": state.get("intent_report", ""),
            "product_fit_report": state.get("product_fit_report", ""),
            "skeptic_report": state.get("skeptic_report", ""),
            "debate_round": state.get("debate_round", 0),
            "case_evidence": self._evidence(state, "case_evidence"),
        }

    def _for_deal_skeptic(self, state: dict[str, Any]) -> dict[str, Any]:
        facts = self._customer_facts(state)
        return {
            "customer": facts,
            "advocate_report": state.get("advocate_report", ""),
            "account_report": state.get("account_report", ""),
            "budget_report": state.get("budget_report", ""),
            "intent_report": state.get("intent_report", ""),
            "product_fit_report": state.get("product_fit_report", ""),
            "debate_round": state.get("debate_round", 0),
            "case_evidence": self._evidence(state, "case_evidence"),
        }

    def _for_strategy_manager(self, state: dict[str, Any]) -> dict[str, Any]:
        facts = self._customer_facts(state)
        return {
            "company": facts["company"],
            "debate_history": list(state.get("debate_history") or []),
            "product_fit_report": state.get("product_fit_report", ""),
            "case_evidence": self._evidence(state, "case_evidence"),
            "sop_evidence": self._evidence(state, "sop_evidence"),
        }

    def _for_sales_manager(self, state: dict[str, Any]) -> dict[str, Any]:
        return {
            "customer": self._customer_facts(state),
            "data_quality_report": state.get("data_quality_report", ""),
            "account_report": state.get("account_report", ""),
            "budget_report": state.get("budget_report", ""),
            "intent_report": state.get("intent_report", ""),
            "product_fit_report": state.get("product_fit_report", ""),
            "strategy_report": state.get("strategy_report", ""),
            "customer_memories": self._memories(state),
            "retrieval_warnings": list(
                state.get("retrieval_warnings") or []
            ),
            "conversation_evidence": self._evidence(
                state, "conversation_evidence"
            ),
            "case_evidence": self._evidence(state, "case_evidence"),
            "sop_evidence": self._evidence(state, "sop_evidence"),
        }
