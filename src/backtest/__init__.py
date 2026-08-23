"""Historical replay harness. Phase 2, not built yet.

The validation gate: at least thirty known winners from 2024 and 2025 plus controls,
replayed with date-bounded OpenAlex queries so author priors are computed as they stood
then, measuring whether the scorer would have put the winners in the top ten on the day
they appeared. Nothing in Phase 2 ships until this is respectable.

The pieces it needs already exist: AuthorPriorService takes a cutoff_date, and features
carry no post-publication information by construction.
"""
