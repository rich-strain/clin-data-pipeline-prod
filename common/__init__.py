"""Shared infrastructure for the Lane 1 (paid) API-calling stages.

`resilient_client` provides the crash-safe cache, retry wrapper, and call-stats
recorder used by Stage 3 extraction (and later Stage 4 synthesize), so those
concerns live in one place rather than being reimplemented per stage.
"""
