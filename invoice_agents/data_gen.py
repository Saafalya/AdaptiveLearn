"""Synthetic data: vendors, POs, goods receipts (GRN) and invoice text with seeded errors.

Every invoice carries a hidden `truth` record, so the PoC can report real accuracy numbers.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, timedelta

from config import TAX_RATE

APPROVED_VENDORS = [
    "Acme Supplies Pvt Ltd", "Bharat Office Mart", "Sahyadri Industrial Tools",
    "Konkan Packaging Co", "Deccan Electronics", "Pune Stationers Ltd",
    "Western Ghats Logistics", "Nashik Hardware Hub",
]
UNKNOWN_VENDORS = ["Zenith Trading Co", "Quickfix Enterprises", "Orbit Supplies LLP"]

CATALOG = {  # sku: (description, list price in INR)
    "SKU-101": ("Laptop Stand", 1250.00),
    "SKU-102": ("Wireless Mouse", 799.00),
    "SKU-103": ("A4 Paper Ream (500)", 285.00),
    "SKU-104": ("Safety Helmet", 640.00),
    "SKU-105": ("Packing Tape Carton", 1980.00),
    "SKU-106": ("LED Panel Light", 1450.00),
    "SKU-107": ("Cordless Drill", 4200.00),
    "SKU-108": ("Network Switch 24p", 3650.00),
    "SKU-109": ("Pallet Wrap Roll", 520.00),
    "SKU-110": ("Ergonomic Chair", 6800.00),
}

ERROR_CYCLE = [
    "PRICE_MISMATCH", "QTY_EXCESS", "DUPLICATE", "UNKNOWN_VENDOR",
    "TAX_ERROR", "PRICE_MISMATCH", "QTY_EXCESS", "DUPLICATE",
]


@dataclass
class Batch:
    invoices: list      # [{"idx", "text", "truth"}]
    vendors: list       # approved vendor master
    po_db: dict         # po_no -> {"vendor", "lines": {sku: {"qty", "price"}}}
    grn_db: dict        # po_no -> {sku: received_qty}
    today: date


def _fmt(x: float, comma: bool) -> str:
    return f"{x:,.2f}" if comma else f"{x:.2f}"


def render_invoice(vendor, inv_no, po_no, inv_date, due_date, lines, tax_rate, comma):
    """Return (invoice_text, total). `comma=True` uses thousand separators (1,250.00)."""
    rows, subtotal = [], 0.0
    for sku, qty, price in lines:
        amount = round(qty * price, 2)
        subtotal += amount
        rows.append(
            f"{sku} | {CATALOG[sku][0]} | Qty {qty} | "
            f"Unit {_fmt(price, comma)} | Amount {_fmt(amount, comma)}"
        )
    subtotal = round(subtotal, 2)
    tax = round(subtotal * tax_rate, 2)
    total = round(subtotal + tax, 2)
    text = "\n".join([
        "TAX INVOICE",
        f"Vendor: {vendor}",
        f"Invoice No: {inv_no}",
        f"PO Number: {po_no}",
        f"Invoice Date: {inv_date}",
        f"Due Date: {due_date}",
        "Items:",
        *rows,
        f"Subtotal: {_fmt(subtotal, comma)}",
        f"Tax ({round(tax_rate * 100)}%): {_fmt(tax, comma)}",
        f"Total: {_fmt(total, comma)}",
    ])
    return text, total


def _build_invoice(i, kind, rng, today, po_db, grn_db):
    vendor = rng.choice(UNKNOWN_VENDORS if kind == "UNKNOWN_VENDOR" else APPROVED_VENDORS)
    inv_no, po_no = f"INV-{7000 + i}", f"PO-{5000 + i}"
    skus = rng.sample(list(CATALOG), rng.randint(1, 3))
    po_lines = {
        s: {"qty": rng.randint(5, 20), "price": round(CATALOG[s][1] * rng.uniform(0.9, 1.1), 2)}
        for s in skus
    }
    received = {s: l["qty"] for s, l in po_lines.items()}
    inv_lines = [[s, l["qty"], l["price"]] for s, l in po_lines.items()]
    tax_rate, expected = TAX_RATE, []

    if kind == "PRICE_MISMATCH":
        j = rng.randrange(len(inv_lines))
        inv_lines[j][2] = round(inv_lines[j][2] * (1 + rng.uniform(0.08, 0.25)), 2)
        expected = ["PRICE_MISMATCH"]
    elif kind == "QTY_EXCESS":
        s = rng.choice(skus)
        received[s] = po_lines[s]["qty"] - rng.randint(1, 3)   # billed more than received
        expected = ["QTY_EXCESS"]
    elif kind == "TAX_ERROR":
        tax_rate = rng.choice([0.22, 0.28])
        expected = ["TAX_ERROR"]
    elif kind == "UNKNOWN_VENDOR":
        po_no = f"PO-{9100 + rng.randint(0, 800)}"              # PO does not exist
        expected = ["NO_PO", "UNKNOWN_VENDOR"]

    if kind != "UNKNOWN_VENDOR":
        po_db[po_no] = {"vendor": vendor, "lines": po_lines}
        grn_db[po_no] = received

    due = today + timedelta(days=rng.randint(-3, 45))
    text, total = render_invoice(
        vendor, inv_no, po_no, (due - timedelta(days=30)).isoformat(), due.isoformat(),
        [tuple(l) for l in inv_lines], tax_rate, comma=rng.random() < 0.4,
    )
    correct = sum(po_lines[s]["price"] * received[s] for s in skus) * (1 + TAX_RATE)
    if kind == "CLEAN":
        at_risk = 0.0
    elif kind == "UNKNOWN_VENDOR":
        at_risk = total
    else:
        at_risk = max(round(total - correct, 2), 0.0)
    return {"text": text, "truth": {
        "kind": kind, "expected_issues": expected, "true_at_risk": at_risk,
        "invoice_no": inv_no, "vendor": vendor, "total": total,
    }}


def generate_batch(n=24, error_pct=0.35, seed=42, today=None) -> Batch:
    rng = random.Random(seed)
    today = today or date.today()
    n = max(n, 5)
    error_pct = min(max(error_pct, 0.0), 0.7)
    n_err = round(n * error_pct)
    error_kinds = [ERROR_CYCLE[i % len(ERROR_CYCLE)] for i in range(n_err)]
    n_dup = error_kinds.count("DUPLICATE")
    base_kinds = [k for k in error_kinds if k != "DUPLICATE"]
    base_kinds += ["CLEAN"] * (n - n_dup - len(base_kinds))
    rng.shuffle(base_kinds)

    po_db, grn_db = {}, {}
    records = [_build_invoice(i, k, rng, today, po_db, grn_db) for i, k in enumerate(base_kinds)]

    clean = [r for r in records if r["truth"]["kind"] == "CLEAN"]
    for _ in range(n_dup):                       # re-submitted copies of earlier clean invoices
        if not clean:
            break
        src = rng.choice(clean)
        dup = {"text": src["text"], "truth": {
            **src["truth"], "kind": "DUPLICATE",
            "expected_issues": ["DUPLICATE"], "true_at_risk": src["truth"]["total"],
        }}
        pos = next(i for i, r in enumerate(records) if r is src)
        records.insert(rng.randint(pos + 1, len(records)), dup)

    for idx, r in enumerate(records):
        r["idx"] = idx                            # arrival order
    return Batch(records, list(APPROVED_VENDORS), po_db, grn_db, today)
