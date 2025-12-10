"""
WBS 3.2: Consensus Algorithm Implementation

Consensus building logic for term extraction pipeline:
- Voting: ≥2/3 model agreement required (models_agreed >= 2 of 3)
- Weighted scoring across models
- Excluded terms tracking with rejection reasons

Patterns Applied:
- Dataclasses for data models (Anti-Pattern #12)
- Pure functions for consensus logic
- Configurable thresholds for flexibility
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ConsensusTerm:
    """A term that reached consensus across models.

    Attributes:
        term: The extracted search term
        score: Weighted score from 0.0 to 1.0
        models_agreed: Number of models that agreed on this term
        model_scores: Individual scores from each model
    """

    term: str
    score: float
    models_agreed: int
    model_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class ExcludedTerm:
    """A term that did not reach consensus.

    Attributes:
        term: The rejected search term
        reason: Why the term was excluded
        stage: Which stage rejected the term
        scores: Individual scores from each model
        models_agreed: Number of models that agreed (below threshold)
    """

    term: str
    reason: str
    stage: str = "consensus"
    scores: dict[str, float] = field(default_factory=dict)
    models_agreed: int = 0


@dataclass
class ConsensusResult:
    """Result of consensus building.

    Attributes:
        final_terms: List of terms that reached consensus
        excluded_terms: List of terms that did not reach consensus
        total_processed: Total number of terms processed
    """

    final_terms: list[ConsensusTerm] = field(default_factory=list)
    excluded_terms: list[ExcludedTerm] = field(default_factory=list)
    total_processed: int = 0


class ConsensusBuilder:
    """Builds consensus from multi-model term extraction results.

    Uses a ≥2/3 agreement threshold by default (2 of 3 models must agree).
    Terms that fail consensus are tracked with rejection reasons.

    Attributes:
        min_agreement: Minimum models required for consensus (default: 2)
        model_weights: Weight for each model's score contribution
        score_threshold: Minimum score for a model to count as "agreeing"
    """

    DEFAULT_WEIGHTS: dict[str, float] = {
        "generator": 1 / 3,
        "validator": 1 / 3,
        "ranker": 1 / 3,
    }

    def __init__(
        self,
        min_agreement: int = 2,
        model_weights: dict[str, float] | None = None,
        score_threshold: float = 0.5,
    ) -> None:
        """Initialize the consensus builder.

        Args:
            min_agreement: Minimum models that must agree (default: 2)
            model_weights: Custom weights for each model (default: equal weights)
            score_threshold: Minimum score for model agreement (default: 0.5)
        """
        self.min_agreement = min_agreement
        self.model_weights = model_weights or self.DEFAULT_WEIGHTS.copy()
        self.score_threshold = score_threshold

    def check_agreement(self, term: dict[str, Any]) -> dict[str, Any]:
        """Check if a term has sufficient model agreement.

        Args:
            term: Dict with term info and model scores/approvals

        Returns:
            Dict with 'agreed' bool and 'models_agreed' count
        """
        models_agreed = self._count_model_agreement(term)

        return {
            "agreed": models_agreed >= self.min_agreement,
            "models_agreed": models_agreed,
        }

    def calculate_weighted_score(self, term: dict[str, Any]) -> float:
        """Calculate weighted score from model scores.

        Args:
            term: Dict with model scores

        Returns:
            Weighted average score
        """
        scores: dict[str, float] = {
            "generator": float(term.get("generator_score", 0.0)),
            "validator": float(term.get("validator_score", 0.0)),
            "ranker": float(term.get("ranker_score", 0.0)),
        }

        total_weight = sum(self.model_weights.values())
        if total_weight == 0:
            return 0.0

        weighted_sum = sum(
            scores[model] * weight
            for model, weight in self.model_weights.items()
            if model in scores
        )

        return weighted_sum / total_weight

    def build_consensus(
        self,
        terms: list[dict[str, Any]],
    ) -> ConsensusResult:
        """Build consensus from multi-model term results.

        Args:
            terms: List of term dicts with model scores/approvals

        Returns:
            ConsensusResult with accepted and excluded terms
        """
        result = ConsensusResult(total_processed=len(terms))

        for term_data in terms:
            term_name = term_data.get("term", "")
            agreement = self.check_agreement(term_data)
            models_agreed = agreement["models_agreed"]

            # Get model scores for tracking
            scores = {
                "generator": term_data.get("generator_score", 0.0),
                "validator": 1.0 if term_data.get("validator_approved") else 0.0,
                "ranker": term_data.get("ranker_score", 0.0),
            }

            # Check agreement threshold
            if not agreement["agreed"]:
                result.excluded_terms.append(
                    ExcludedTerm(
                        term=term_name,
                        reason=f"insufficient_agreement: {models_agreed}/{self.min_agreement}",
                        stage="consensus",
                        scores=scores,
                        models_agreed=models_agreed,
                    )
                )
                continue

            # Calculate weighted score
            weighted_score = self.calculate_weighted_score(term_data)

            # Term reached consensus
            result.final_terms.append(
                ConsensusTerm(
                    term=term_name,
                    score=weighted_score,
                    models_agreed=models_agreed,
                    model_scores=scores,
                )
            )

        # Sort by score descending
        result.final_terms = sorted(
            result.final_terms, key=lambda t: t.score, reverse=True
        )

        return result

    def _count_model_agreement(self, term: dict[str, Any]) -> int:
        """Count how many models agree on a term."""
        count = 0

        # Generator agrees if score >= threshold
        gen_score = term.get("generator_score", 0.0)
        if gen_score >= self.score_threshold:
            count += 1

        # Validator agrees if approved
        if term.get("validator_approved", False):
            count += 1

        # Ranker agrees if score >= threshold
        ranker_score = term.get("ranker_score", 0.0)
        if ranker_score >= self.score_threshold:
            count += 1

        return count
