"""LLM backends. MockLLM needs nothing; OllamaLLM talks to a local Ollama server."""
from __future__ import annotations

import json
import re

from schemas import InvoiceExtract


def template_explanation(p: dict) -> str:
    if p["decision"] == "APPROVED":
        return ("All checks passed (vendor, PO, GRN quantities, prices, tax, duplicate). "
                "Safe to schedule for payment.")
    reasons = "; ".join(i["detail"] for i in p["issues"]) or "extraction could not be validated"
    who = "Escalated to a finance reviewer" if p["decision"] == "ESCALATED" else "Held for vendor correction"
    return f"{who}: {reasons}. Amount at risk: INR {p['at_risk']:,.2f}."


class MockLLM:
    name = "mock (rule-based, no LLM)"
    can_extract = False

    def extract(self, text: str):
        return None

    def explain(self, payload: dict) -> str:
        return template_explanation(payload)


class OllamaLLM:
    can_extract = True

    def __init__(self, model: str = "llama3.2"):
        from langchain_ollama import ChatOllama
        self.name = f"ollama:{model}"
        self.chat = ChatOllama(model=model, temperature=0)
        self.json_chat = ChatOllama(model=model, temperature=0, format="json")

    def ping(self):
        self.chat.invoke("Reply with the single word: ok")

    def extract(self, text: str):
        prompt = (
            "Extract this invoice into JSON with exactly these keys: invoice_no, vendor, "
            "po_number, invoice_date, due_date, subtotal, tax, tax_rate (percent, e.g. 18), "
            "total, lines (list of {sku, description, qty, unit_price, amount}). "
            "Use plain numbers without thousands separators. Return JSON only.\n\nINVOICE:\n" + text
        )
        try:
            data = json.loads(self.json_chat.invoke(prompt).content)
            data["tax_rate"] = float(data.get("tax_rate", 0)) / 100
            return InvoiceExtract(**data)
        except Exception:
            return None            # the Extraction Agent falls back to its parser

    def explain(self, payload: dict) -> str:
        facts = template_explanation(payload)
        prompt = (
            "You are an accounts-payable assistant. Rewrite the finding below as 1-2 clear "
            "sentences for a finance reviewer. Do not add facts or numbers.\n\n" + facts
        )
        try:
            out = self.chat.invoke(prompt).content.strip()
            return re.sub(r"\s+", " ", out) or facts
        except Exception:
            return facts


def get_llm(engine: str, model: str = "llama3.2"):
    """Return (llm, warning_or_None). Falls back to the mock if Ollama is unavailable."""
    if engine != "ollama":
        return MockLLM(), None
    try:
        llm = OllamaLLM(model)
        llm.ping()
        return llm, None
    except Exception as e:  # noqa: BLE001
        return MockLLM(), f"Could not reach Ollama model '{model}' ({type(e).__name__}). Using the mock LLM."
