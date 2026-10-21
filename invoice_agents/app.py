"""Streamlit PoC: agentic three-way invoice matching for accounts payable.

Run with:  streamlit run app.py
"""
import time
from collections import Counter

import pandas as pd
import streamlit as st

from config import APPROVED, ESCALATED, HOLD
from data_gen import generate_batch
from llm import get_llm
from orchestrator import HAS_LANGGRAPH, Ctx, evaluate, process_batch

st.set_page_config(page_title="AP Three-Way Match Agents", page_icon="🧾", layout="wide")

COLORS = {APPROVED: "#d1fae5", HOLD: "#fef3c7", ESCALATED: "#fee2e2"}

ARCHITECTURE_DOT = """
digraph G {
  rankdir=LR; nodesep=0.4; ranksep=0.5;
  node [shape=box, style="rounded,filled", fillcolor="#eef2ff", color="#6366f1", fontname="Helvetica", fontsize=11];
  edge [color="#64748b", fontname="Helvetica", fontsize=9];

  batch   [label="Invoice batch\\n(text from email / OCR)", shape=folder, fillcolor="#f1f5f9", color="#94a3b8"];
  planner [label="Planner Agent\\nprioritise + sequence"];
  extract [label="Extraction Agent\\nfast parser / LLM -> schema"];
  critic1 [label="Critic Agent\\nvalidate extraction", fillcolor="#fff7ed", color="#f97316"];
  matcher [label="Matching Agent\\n3-way match tools"];
  critic2 [label="Critic Agent\\ncross-check variance", fillcolor="#fff7ed", color="#f97316"];
  decider [label="Decision Agent\\napprove / hold / escalate"];
  out     [label="Payment run  |  Vendor query  |  Human review queue", shape=note, fillcolor="#ecfdf5", color="#10b981"];
  data    [label="Tools and data\\nPO DB - GRN DB - Vendor master\\nDuplicate registry - Business rules", shape=cylinder, fillcolor="#f1f5f9", color="#94a3b8"];
  llm     [label="LLM (Ollama or mock)\\nextraction + explanations", shape=ellipse, fillcolor="#faf5ff", color="#a855f7"];

  batch -> planner -> extract -> critic1;
  critic1 -> extract [label="rejected: retry\\n(robust parser)", style=dashed, color="#f97316"];
  critic1 -> matcher [label="valid"];
  critic1 -> decider [label="invalid after retry", style=dashed];
  matcher -> critic2 -> decider -> out;
  data -> matcher [style=dotted, dir=both];
  llm -> extract [style=dotted];
  llm -> decider [style=dotted];
}
"""


# --------------------------------------------------------------------------- helpers
def timed(gen):
    """Yield (event, seconds spent inside the generator) so UI sleeps are not counted."""
    it = iter(gen)
    while True:
        t0 = time.perf_counter()
        try:
            ev = next(it)
        except StopIteration:
            return
        yield ev, time.perf_counter() - t0


def results_df(results):
    return pd.DataFrame([{
        "#": r["idx"], "Invoice": r["invoice_no"], "Vendor": r["vendor"],
        "Total (INR)": r["total"], "Priority": r["priority"], "Decision": r["decision"],
        "Issues": ", ".join(sorted({i["type"] for i in r["issues"]})) or "-",
        "At risk (INR)": r["at_risk"], "Attempts": r["attempts"],
    } for r in results])


def style(df):
    fmt = {c: "{:,.2f}" for c in ("Total (INR)", "At risk (INR)") if c in df.columns}
    return (df.style
            .apply(lambda row: [f"background-color: {COLORS[row['Decision']]}; color: #111"] * len(row), axis=1)
            .format(fmt))


# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Batch")
    n_inv = st.slider("Invoices in batch", 10, 60, 24)
    err_pct = st.slider("Seeded error rate", 0.10, 0.60, 0.35, 0.05)
    seed = st.number_input("Random seed", 0, 9999, 42)
    regenerate = st.button("Generate new batch")

    st.header("Agent engine")
    engine = st.radio("LLM backend", ["Mock (no setup needed)", "Ollama (local model)"])
    model = st.text_input("Ollama model", "llama3.2") if engine.startswith("Ollama") else ""
    use_graph = st.checkbox("Orchestrate with LangGraph", value=HAS_LANGGRAPH, disabled=not HAS_LANGGRAPH)
    if not HAS_LANGGRAPH:
        st.caption("langgraph is not installed, using the built-in runner (same workflow).")
    delay = st.slider("Demo delay per step (s)", 0.0, 0.5, 0.08, 0.02)

    st.header("Time-saving assumptions")
    manual_min = st.number_input("Manual review, min per invoice", 1.0, 30.0, 6.0, 0.5)
    review_min = st.number_input("Human review of a flagged invoice, min", 1.0, 30.0, 4.0, 0.5)

if regenerate or "batch" not in st.session_state:
    st.session_state.batch = generate_batch(n_inv, err_pct, int(seed))
    st.session_state.pop("results", None)

batch = st.session_state.batch

st.title("🧾 Invoice Three-Way Match Agents")
st.caption("Multi-agent accounts-payable workflow: plan, extract, validate, match, decide. All data is synthetic.")

tab_run, tab_results, tab_arch = st.tabs(["1. Run agents", "2. Results and validation", "3. Architecture"])

# --------------------------------------------------------------------------- tab 1: run
with tab_run:
    st.subheader("Incoming batch")
    st.write("Each invoice must agree with its purchase order (what we ordered) and goods receipt note "
             "(what actually arrived) before it is paid. Some invoices in this batch contain seeded errors.")
    reveal = st.checkbox("Reveal answer key (seeded errors)")
    preview = pd.DataFrame([{
        "#": i["idx"], "Invoice": i["truth"]["invoice_no"], "Vendor": i["truth"]["vendor"],
        "Total (INR)": i["truth"]["total"],
        **({"Seeded error": i["truth"]["kind"]} if reveal else {}),
    } for i in batch.invoices])
    st.dataframe(preview, use_container_width=True, hide_index=True, height=240)
    with st.expander("Master data the agents look up (POs, goods receipts, vendors)"):
        st.dataframe(pd.DataFrame([
            {"PO": po, "Vendor": d["vendor"], "SKU": sku, "Ordered": l["qty"], "PO price": l["price"],
             "Received (GRN)": batch.grn_db[po].get(sku)}
            for po, d in batch.po_db.items() for sku, l in d["lines"].items()
        ]), use_container_width=True, hide_index=True)
        st.write("Approved vendors:", ", ".join(batch.vendors))
    with st.expander("Sample invoice text"):
        st.code(batch.invoices[0]["text"], language=None)

    if st.button("▶ Run agents", type="primary"):
        llm, warn = get_llm("ollama" if engine.startswith("Ollama") else "mock", model)
        if warn:
            st.warning(warn)
        ctx = Ctx(llm, batch.vendors, batch.po_db, batch.grn_db, batch.today)
        total = len(batch.invoices)
        bar = st.progress(0.0, text="Planner Agent is sequencing the batch...")
        left, right = st.columns([3, 2])
        left.markdown("**Live agent trace**")
        trace_box = left.empty()
        right.markdown("**Decisions so far**")
        table_box = right.empty()

        lines, results, proc = [], [], 0.0
        for ev, dt in timed(process_batch(batch, ctx, use_graph)):
            proc += dt
            if ev["type"] == "plan":
                entries, prefix = ev["log"], ""
            elif ev["type"] == "step":
                entries, prefix = ev["update"].get("log", []), f"#{ev['idx']:02d} "
            else:
                entries, prefix = [], ""
            lines += [f"{prefix}{e['agent']}: {e['msg']}" for e in entries]
            trace_box.code("\n".join(lines[-16:]), language=None)
            if ev["type"] == "result":
                results.append(ev["result"])
                bar.progress(len(results) / total, text=f"{len(results)}/{total} invoices processed")
                table_box.dataframe(
                    style(results_df(results)[["Invoice", "Decision", "At risk (INR)"]].iloc[::-1]),
                    use_container_width=True, hide_index=True, height=300)
            elif ev["type"] == "step":
                time.sleep(delay)

        st.session_state.results = results
        st.session_state.proc_seconds = proc
        st.session_state.engine_name = llm.name
        bar.progress(1.0, text="Done")
        st.success("Batch complete. Open the 'Results and validation' tab.")

# --------------------------------------------------------------------------- tab 2: results
with tab_results:
    res = st.session_state.get("results")
    if not res:
        st.info("Run the agents first (tab 1).")
    else:
        m = evaluate(res)
        counts = Counter(r["decision"] for r in res)
        st.caption(f"Engine: {st.session_state.engine_name}")

        c = st.columns(4)
        c[0].metric("Invoices processed", m["n"])
        c[1].metric("Approved", counts[APPROVED])
        c[2].metric("Held", counts[HOLD])
        c[3].metric("Escalated", counts[ESCALATED])

        st.subheader("Validation against the seeded answer key")
        v = st.columns(4)
        v[0].metric("Seeded errors caught", f"{m['caught']}/{m['seeded']}", f"{m['recall']:.0%} recall", delta_color="off")
        v[1].metric("Correct issue type found", f"{m['type_correct']}/{m['seeded']}")
        v[2].metric("Clean invoices wrongly flagged", f"{m['false_positives']}/{m['clean']}")
        v[3].metric("Bad extractions fixed by Critic loop", f"{m['recovered_by_retry']}/{m['retried']}")

        st.subheader("Money and time")
        flagged = m["n"] - counts[APPROVED]
        manual_total = m["n"] * manual_min
        assisted = st.session_state.proc_seconds / 60 + flagged * review_min
        saved = manual_total - assisted
        w = st.columns(4)
        w[0].metric("Payment value blocked (INR)", f"{m['blocked_value']:,.0f}")
        w[1].metric("Overpayment risk detected (INR)", f"{m['detected_at_risk']:,.0f}",
                    f"true seeded: {m['true_at_risk']:,.0f}", delta_color="off")
        w[2].metric("Manual effort (min)", f"{manual_total:,.0f}")
        w[3].metric("Agent-assisted effort (min)", f"{assisted:,.0f}",
                    f"{saved / manual_total:.0%} saved" if manual_total else None, delta_color="off")
        st.caption("Effort figures are estimates from the sidebar assumptions. Agent compute time is measured "
                   "(demo delay excluded); humans only review flagged invoices.")

        ch1, ch2 = st.columns(2)
        ch1.markdown("**Decisions**")
        ch1.bar_chart(pd.Series(counts, name="invoices"))
        types = Counter(i["type"] for r in res for i in r["issues"])
        ch2.markdown("**Issues found by type**")
        if types:
            ch2.bar_chart(pd.Series(types, name="issues"))

        st.subheader("All decisions")
        pick = st.multiselect("Filter by decision", [APPROVED, HOLD, ESCALATED], default=[APPROVED, HOLD, ESCALATED])
        df = results_df(sorted(res, key=lambda r: r["idx"]))
        df = df[df["Decision"].isin(pick)]
        st.dataframe(style(df), use_container_width=True, hide_index=True)
        st.download_button("Download results (CSV)", df.to_csv(index=False), "ap_results.csv", "text/csv")

        st.subheader("Inspect one invoice")
        ordered = sorted(res, key=lambda r: r["idx"])
        r = st.selectbox("Invoice", ordered,
                         format_func=lambda r: f"#{r['idx']:02d}  {r['invoice_no']}  -  {r['vendor']}  [{r['decision']}]")
        a, b = st.columns(2)
        a.markdown("**Raw invoice**")
        a.code(r["raw"], language=None)
        b.markdown("**Decision**")
        b.info(f"{r['decision']}: {r['reason']}")
        if r["issues"]:
            b.dataframe(pd.DataFrame(r["issues"]), use_container_width=True, hide_index=True)
        b.markdown("**Agent trace**")
        b.code("\n".join(f"{e['agent']}: {e['msg']}" for e in r["trace"]), language=None)
        with st.expander("Extracted data (validated schema)"):
            st.json(r["extract"] or {})
        st.caption(f"Answer key: {r['truth']['kind']}")

# --------------------------------------------------------------------------- tab 3: architecture
with tab_arch:
    st.subheader("Agent workflow")
    st.graphviz_chart(ARCHITECTURE_DOT, use_container_width=True)
    st.markdown("""
| Agent | Responsibility | Tools it uses |
|---|---|---|
| **Planner** | Sequences the batch by due date and amount, flags urgent invoices | header peek |
| **Extraction** | Turns invoice text into a typed schema (LLM or parser); retries with a robust parser if rejected | `parse_invoice`, LLM |
| **Critic** | (1) Rejects extractions whose arithmetic does not add up. (2) Checks every money variance is explained by a named issue | `validate_extraction` |
| **Matching** | Three-way match of invoice, PO and GRN; price, quantity, tax, vendor and duplicate checks | `three_way_match`, PO/GRN DB, duplicate registry |
| **Decision** | Applies business rules, then writes a plain-language reason | `decide`, LLM |

**Decision rules:** no issues -> *approved*. Duplicate, unknown vendor, missing PO, vendor mismatch or unexplained variance -> *escalated*.
Other mismatches -> *held* for vendor correction, or *escalated* if the amount at risk is 10,000 INR or more.
""")
