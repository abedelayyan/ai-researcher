"""Day-zero feature extraction.

Nothing in this package may import from src.outcomes or read an outcome table. The
rule is enforced by tests/test_no_leakage.py and by the scoped SQLite connection in
src.store.db.open_feature_scoped.
"""
