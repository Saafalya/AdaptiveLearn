"""Structured data the agents pass around."""
from typing import List
from pydantic import BaseModel


class LineItem(BaseModel):
    sku: str
    description: str
    qty: int
    unit_price: float
    amount: float


class InvoiceExtract(BaseModel):
    invoice_no: str
    vendor: str
    po_number: str
    invoice_date: str
    due_date: str
    lines: List[LineItem]
    subtotal: float
    tax: float
    tax_rate: float      # fraction, e.g. 0.18
    total: float
