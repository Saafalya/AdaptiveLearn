"""The five agents. Each node takes (state, ctx) and returns a partial state update."""
from __future__ import annotations

from config import MAX_EXTRACT_ATTEMPTS, URGENT_DAYS, VARIANCE_TOLERANCE
from tools import decide, parse_invoice, peek_header, three_way_match, validate_extraction

EXPLAINED_BY = {"PRICE_MISMATCH", "QTY_EXCESS", "TAX_ERROR", "UNORDERED_ITEM", "DUPLICATE",
                "NO_PO", "UNKNOWN_VENDOR", "VENDOR_MISMATCH"}


def _log(agent: str, msg: str) -> dict:
    return {"agent": agent, "msg": msg}


# --------------------------------------------------------------------------- Planner (batch level)
def planner_agent(raws: list[str], today) -> list[dict]:
    """Sequence the batch: nearest due date first, then biggest amount, then arrival order."""
    plan = []
    for idx, raw in enumerate(raws):
        h = peek_header(raw, today)
        urgent = h["days_left"] <= URGENT_DAYS
        plan.append({
            "idx": idx, **h,
            "priority": "URGENT" if urgent else "NORMAL",
            "checks": ["extract", "validate", "3-way match", "duplicate", "tax"],
        })
    plan.sort(key=lambda p: (p["days_left"], -p["total_hint"], p["idx"]))
    return plan


# --------------------------------------------------------------------------- Extraction Agent
def extractor_agent(state, ctx):
    attempt = state.get("attempt", 0) + 1
    name = "Extraction Agent"
    ext, method = None, ""
    if attempt == 1 and ctx.llm.can_extract:
        ext, method = ctx.llm.extract(state["raw"]), f"LLM ({ctx.llm.name})"
        if ext is None:
            method = "LLM returned invalid JSON -> fast parser"
    if ext is None:
        robust = attempt > 1
        method = method or ("robust parser (handles thousand separators)" if robust else "fast parser")
        try:
            ext = parse_invoice(state["raw"], robust=robust)
        except Exception as e:  # noqa: BLE001
            return {"attempt": attempt, "extract": None,
                    "log": [_log(name, f"attempt {attempt} via {method}: parse failed ({e})")]}
    return {"attempt": attempt, "extract": ext,
            "log": [_log(name, f"attempt {attempt} via {method}: {ext.invoice_no} | {ext.vendor} | total {ext.total:,.2f}")]}


# --------------------------------------------------------------------------- Critic Agent (1)
def critic_extract_agent(state, ctx):
    name, ext = "Critic Agent", state.get("extract")
    problems = ["extraction produced no result"] if ext is None else validate_extraction(ext)
    if not problems:
        return {"extract_ok": True, "critic_issues": [],
                "log": [_log(name, "extraction PASSED arithmetic and format checks")]}
    if state["attempt"] < MAX_EXTRACT_ATTEMPTS:
        msg = f"extraction REJECTED ({problems[0]}) -> asking Extraction Agent to retry"
    else:
        msg = f"extraction REJECTED again ({problems[0]}) -> cannot trust data, escalating"
    return {"extract_ok": False, "critic_issues": problems, "log": [_log(name, msg)]}


def route_after_extract_critic(state) -> str:
    if state["extract_ok"]:
        return "matcher"
    return "extractor" if state["attempt"] < MAX_EXTRACT_ATTEMPTS else "decider"


# --------------------------------------------------------------------------- Matching Agent
def matcher_agent(state, ctx):
    inv = state["extract"]
    issues, expected_total, at_risk = three_way_match(inv, ctx.vendors, ctx.po_db, ctx.grn_db, ctx.seen)
    msg = (f"{len(issues)} issue(s): " + ", ".join(sorted({i['type'] for i in issues}))
           if issues else "invoice, PO and GRN agree")
    return {"issues": issues, "expected_total": expected_total, "at_risk": at_risk,
            "log": [_log("Matching Agent", f"3-way match done -> {msg}")]}


# --------------------------------------------------------------------------- Critic Agent (2)
def critic_match_agent(state, ctx):
    """Cross-check: any money variance must be explained by at least one named issue."""
    inv, issues = state["extract"], list(state["issues"])
    variance = round(inv.total - state["expected_total"], 2)
    types = {i["type"] for i in issues}
    if abs(variance) > VARIANCE_TOLERANCE and not (types & EXPLAINED_BY):
        issues.append({"type": "UNEXPLAINED_VARIANCE",
                       "detail": f"total differs from PO/GRN expectation by {variance:,.2f} with no matching rule"})
        msg = f"variance {variance:,.2f} is unexplained -> added UNEXPLAINED_VARIANCE"
    else:
        msg = f"variance {variance:,.2f} is consistent with findings"
    return {"issues": issues, "log": [_log("Critic Agent", msg)]}


# --------------------------------------------------------------------------- Decision Agent
def decision_agent(state, ctx):
    ext, issues = state.get("extract"), state.get("issues", [])
    at_risk = state.get("at_risk", ext.total if ext else 0.0)
    decision, rule = decide(issues, at_risk, state.get("extract_ok", False))
    reason = ctx.llm.explain({
        "decision": decision, "issues": issues, "at_risk": at_risk,
        "invoice_no": ext.invoice_no if ext else "?", "total": ext.total if ext else 0.0,
    })
    return {"decision": decision, "reason": reason, "at_risk": at_risk,
            "log": [_log("Decision Agent", f"{decision} ({rule})")]}
