"""Limits applied to public API requests."""

MAX_CELLS = 12_000  # largest layout accepted (BFS runs once per item)
MAX_ITEMS = 60  # items per trip
MAX_ORDERS = 200  # trips per bulk request
MAX_UPLOAD_BYTES = 1_000_000  # CSV upload size
