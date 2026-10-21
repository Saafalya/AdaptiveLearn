"""Central settings for the AP three-way-match PoC."""

TAX_RATE = 0.18               # GST rate the company expects on every invoice
PRICE_TOLERANCE = 0.01        # INR; allowed unit-price difference vs PO
VARIANCE_TOLERANCE = 1.00     # INR; allowed total difference vs expected
ESCALATE_AT_RISK = 10_000.0   # INR; mismatches above this go to a human
MAX_EXTRACT_ATTEMPTS = 2      # Extraction Agent retries once after Critic rejection
URGENT_DAYS = 7               # invoices due within this many days are fast-tracked

APPROVED, HOLD, ESCALATED = "APPROVED", "HOLD", "ESCALATED"

# Issue types that always need a human, whatever the amount (fraud / control risks)
ALWAYS_ESCALATE = {
    "DUPLICATE", "UNKNOWN_VENDOR", "NO_PO", "VENDOR_MISMATCH", "UNEXPLAINED_VARIANCE",
}
