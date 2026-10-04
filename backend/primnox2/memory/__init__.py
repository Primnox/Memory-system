"""Permanent memory — V2.2's other half.

The knowledge graph knows about files; this knows about the user. "Where is
Stripe used" is the graph; "I prefer dark mode" is here.

The distinction is worth keeping sharp, because the failure mode of blurring it
is a system that quietly turns a passing remark into a permanent fact.
"""
from . import service  # noqa: F401
