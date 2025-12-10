"""Model agents for keyword extraction, validation, and ranking.

Agents:
- CodeT5Agent: Generator - extracts terms from text
- GraphCodeBERTAgent: Validator - filters generic terms
- CodeBERTAgent: Ranker - scores and ranks by similarity

Patterns applied from CODING_PATTERNS_ANALYSIS.md:
- Repository Pattern with Protocol (Phase 2)
- FakeClient for testing (Anti-Pattern #12)
"""
