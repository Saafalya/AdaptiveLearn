"""Workflow wiring: LangGraph if installed, otherwise an equivalent plain-Python runner."""
from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, Any, Optional, TypedDict

from agents import (critic_extract_agent, critic_match_agent, decision_agent, extractor_agent,
                    matcher_agent, planner_agent, route_after_extract_critic)
from config import APPROVED

try:
    from langgraph.graph import END, StateGraph
    HAS_LANGGRAPH = True
except ImportError:  # the PoC still runs, just without LangGraph
    HAS_LANGGRAPH = False


class State(TypedDict, total=False):
    raw: str
    plan: dict
    attempt: int
    extract: Any
    extract_ok: bool
    critic_issues: list
    issues: list
    expected_total: float
    at_risk: float
    decision: str
    reason: str
    log: Annotated[list, operator.add]


@dataclass
class Ctx:
    llm: Any
    vendors: list
    po_db: dict
    grn_db: dict
    today: Any
    seen: set = field(default_factory=set)      # duplicate registry for this run


NODES = {
    "extractor": extractor_agent,
    "critic_extract": critic_extract_agent,
    "matcher": matcher_agent,
    "critic_match": critic_match_agent,
    "decider": decision_agent,
}


def build_graph(ctx: Ctx):
    g = StateGraph(State)
    def bind(fn):
        return lambda state: fn(state, ctx)

    for name, fn in NODES.items():
        g.add_node(name, bind(fn))
    g.set_entry_point("extractor")
    g.add_edge("extractor", "critic_extract")
    g.add_conditional_edges("critic_extract", route_after_extract_critic,
                            {"extractor": "extractor", "matcher": "matcher", "decider": "decider"})
    g.add_edge("matcher", "critic_match")
    g.add_edge("critic_match", "decider")
    g.add_edge("decider", END)
    return g.compile()


def _merge(state: dict, upd: dict) -> dict:
    new = dict(state)
    for k, v in upd.items():
        new[k] = list(state.get("log", [])) + list(v) if k == "log" else v
    return new


def _next(node: str, state: dict) -> Optional[str]:
    """Plain-Python mirror of the LangGraph edges above."""
    if node == "critic_extract":
        return route_after_extract_critic(state)
    return {"extractor": "critic_extract", "matcher": "critic_match",
            "critic_match": "decider", "decider": None}[node]


def run_invoice(raw: str, plan_item: dict, ctx: Ctx, graph=None):
    """Yield (node, update, state_so_far) for each agent step on one invoice."""
    state: dict = {"raw": raw, "plan": plan_item, "attempt": 0, "log": []}
    if graph is not None:
        for chunk in graph.stream(state, stream_mode="updates"):
            for node, upd in chunk.items():
                state = _merge(state, upd)
                yield node, upd, state
    else:
        node: Optional[str] = "extractor"
        while node:
            upd = NODES[node](state, ctx)
            state = _merge(state, upd)
            yield node, upd, state
            node = _next(node, state)


def _to_result(inv: dict, item: dict, state: dict) -> dict:
    ext = state.get("extract")
    return {
        "idx": item["idx"], "priority": item["priority"], "days_left": item["days_left"],
        "invoice_no": ext.invoice_no if ext else "(unreadable)",
        "vendor": ext.vendor if ext else "(unreadable)",
        "total": ext.total if ext else item["total_hint"],
        "decision": state["decision"], "reason": state["reason"],
        "issues": state.get("issues", []), "critic_issues": state.get("critic_issues", []),
        "at_risk": round(state.get("at_risk", 0.0), 2), "attempts": state.get("attempt", 1),
        "extract": ext.model_dump() if ext else None,
        "raw": inv["text"], "trace": state["log"], "truth": inv["truth"],
    }


def process_batch(batch, ctx: Ctx, use_langgraph: bool = True):
    """Generator of UI events: plan -> (step* -> result)*."""
    plan = planner_agent([i["text"] for i in batch.invoices], ctx.today)
    n_urgent = sum(p["priority"] == "URGENT" for p in plan)
    yield {"type": "plan", "plan": plan, "log": [{
        "agent": "Planner Agent",
        "msg": f"sequenced {len(plan)} invoices by due date and amount; {n_urgent} urgent (due within 7 days)"}]}
    graph = build_graph(ctx) if (use_langgraph and HAS_LANGGRAPH) else None
    for pos, item in enumerate(plan):
        inv, state = batch.invoices[item["idx"]], {}
        for node, upd, state in run_invoice(batch.invoices[item["idx"]]["text"], item, ctx, graph):
            yield {"type": "step", "pos": pos, "idx": item["idx"], "node": node, "update": upd}
        yield {"type": "result", "result": _to_result(inv, item, state)}


def evaluate(results: list[dict]) -> dict:
    """Compare agent decisions with the hidden seeded ground truth."""
    seeded = [r for r in results if r["truth"]["expected_issues"]]
    clean = [r for r in results if not r["truth"]["expected_issues"]]
    caught = [r for r in seeded if r["decision"] != APPROVED]
    exact = [r for r in seeded
             if set(r["truth"]["expected_issues"]) <= {i["type"] for i in r["issues"]}]
    false_pos = [r for r in clean if r["decision"] != APPROVED]
    return {
        "n": len(results), "seeded": len(seeded), "caught": len(caught),
        "recall": len(caught) / len(seeded) if seeded else 1.0,
        "type_correct": len(exact), "clean": len(clean), "false_positives": len(false_pos),
        "retried": sum(r["attempts"] > 1 for r in results),
        "recovered_by_retry": sum(r["attempts"] > 1 and r["extract"] is not None
                                  and not r["critic_issues"] for r in results),
        "true_at_risk": sum(r["truth"]["true_at_risk"] for r in seeded),
        "detected_at_risk": sum(r["at_risk"] for r in results if r["decision"] != APPROVED),
        "blocked_value": sum(r["total"] for r in results if r["decision"] != APPROVED),
    }
