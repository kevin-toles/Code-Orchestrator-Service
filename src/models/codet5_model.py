"""
Code-Orchestrator-Service - CodeT5+ Model Wrapper

Full-featured CodeT5+ model wrapper exposing all capabilities:
- Code Summarization
- Code Generation
- Code Translation
- Code Completion
- Code Understanding
- Defect Detection
- Clone Detection

Uses locally hosted Salesforce/codet5p-220m with singleton pattern.

Architecture Role: GENERATOR (Sous Chef)
Reference: KITCHEN_BRIGADE_ARCHITECTURE.md
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

from src.core.logging import get_logger

logger = get_logger(__name__)

# Local model path
_LOCAL_MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "codet5"
_HUGGINGFACE_MODEL = "Salesforce/codet5p-220m"

# Singleton instance
_model_instance: CodeT5Model | None = None


def get_codet5_model() -> CodeT5Model:
    """Get or create singleton CodeT5+ model instance.

    Returns:
        Shared CodeT5Model instance

    Raises:
        RuntimeError: If model fails to load
    """
    global _model_instance
    if _model_instance is None:
        _model_instance = CodeT5Model()
    return _model_instance


class CodeT5Model:
    """CodeT5+ model wrapper with full capabilities.

    Supports all CodeT5+ use cases:
    - summarize(): Code → Natural Language
    - generate(): Natural Language → Code
    - translate(): Code (Lang A) → Code (Lang B)
    - complete(): Code Prefix → Code Continuation
    - understand(): Code → Semantic Analysis
    - detect_defects(): Code → Bug Detection
    - detect_clones(): Code → Clone Detection

    Uses singleton pattern to avoid loading model multiple times.
    """

    # Language detection patterns
    LANGUAGE_PATTERNS = {
        "python": [r"def\s+\w+\s*\(", r"import\s+\w+", r"from\s+\w+\s+import", r":\s*$", r"self\.", r"__init__"],
        "javascript": [r"function\s+\w+", r"const\s+\w+\s*=", r"let\s+\w+\s*=", r"=>\s*{", r"require\(", r"module\.exports"],
        "typescript": [r"interface\s+\w+", r":\s*(string|number|boolean)", r"<\w+>", r"export\s+(class|interface|type)"],
        "java": [r"public\s+(class|void|static)", r"private\s+\w+", r"System\.out", r"@Override", r"extends\s+\w+"],
        "cpp": [r"#include\s*<", r"std::", r"int\s+main\s*\(", r"cout\s*<<", r"nullptr", r"->"],
        "csharp": [r"using\s+System", r"namespace\s+\w+", r"public\s+class", r"Console\.Write", r"\[.*\]"],
        "go": [r"package\s+\w+", r"func\s+\w+\s*\(", r"fmt\.", r":=", r"import\s+\("],
        "rust": [r"fn\s+\w+\s*\(", r"let\s+mut", r"impl\s+\w+", r"pub\s+fn", r"->", r"::"],
        "ruby": [r"def\s+\w+", r"end\s*$", r"puts\s+", r"require\s+['\"]", r"attr_\w+"],
        "php": [r"<\?php", r"\$\w+", r"function\s+\w+", r"echo\s+", r"->"],
    }

    def __init__(self) -> None:
        """Initialize CodeT5+ model and tokenizer."""
        self._tokenizer: Any = None
        self._model: T5ForConditionalGeneration | None = None
        self._device: str = "cpu"
        self._loaded: bool = False
        self._load_model()

    def _load_model(self) -> None:
        """Load CodeT5+ model and tokenizer."""
        model_path = str(_LOCAL_MODEL_PATH)

        if not _LOCAL_MODEL_PATH.exists():
            model_path = _HUGGINGFACE_MODEL
            logger.warning("local_codet5_not_found", path=str(_LOCAL_MODEL_PATH), using=model_path)
        else:
            logger.info("loading_codet5_from_local", path=model_path)

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self._model = T5ForConditionalGeneration.from_pretrained(model_path, trust_remote_code=True)

            # Use GPU if available
            self._device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            self._model = self._model.to(self._device)
            self._loaded = True

            logger.info("codet5_model_loaded", device=self._device, model=model_path)

        except Exception as e:
            logger.error("codet5_model_load_failed", error=str(e))
            raise RuntimeError(f"Failed to load CodeT5+ model: {e}") from e

    @property
    def device(self) -> str:
        """Get the device the model is running on."""
        return self._device

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    def _detect_language(self, code: str) -> str:
        """Auto-detect programming language from code.

        Args:
            code: Source code to analyze

        Returns:
            Detected language name or 'unknown'
        """
        scores: dict[str, int] = {}

        for lang, patterns in self.LANGUAGE_PATTERNS.items():
            score = sum(1 for p in patterns if re.search(p, code, re.MULTILINE))
            if score > 0:
                scores[lang] = score

        if not scores:
            return "unknown"

        return max(scores, key=lambda k: scores[k])

    def _generate(
        self,
        prompt: str,
        max_length: int = 128,
        num_beams: int = 4,
        num_return_sequences: int = 1,
    ) -> list[str]:
        """Generate text using CodeT5+.

        Args:
            prompt: Input prompt
            max_length: Maximum output length
            num_beams: Beam search width
            num_return_sequences: Number of sequences to return

        Returns:
            List of generated texts
        """
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
                early_stopping=True,
                do_sample=False,
            )

        return self._tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def _get_embeddings(self, texts: list[str]) -> torch.Tensor:
        """Get encoder embeddings for texts.

        Args:
            texts: List of texts to embed

        Returns:
            Tensor of embeddings
        """
        embeddings = []

        for text in texts:
            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True,
            ).to(self._device)

            with torch.no_grad():
                encoder_outputs = self._model.encoder(**inputs)
                # Mean pooling over sequence length
                embedding = encoder_outputs.last_hidden_state.mean(dim=1)
                embeddings.append(embedding)

        return torch.cat(embeddings, dim=0)

    # =========================================================================
    # PUBLIC API - All CodeT5+ Capabilities
    # =========================================================================

    def summarize(
        self,
        code: str,
        language: str | None = None,
        max_length: int = 128,
    ) -> dict[str, Any]:
        """Generate natural language summary of code.

        Args:
            code: Source code to summarize
            language: Programming language (auto-detected if None)
            max_length: Maximum summary length

        Returns:
            Dict with summary, language, confidence
        """
        detected_lang = language or self._detect_language(code)
        prompt = f"Summarize this {detected_lang} code: {code}"

        generated = self._generate(prompt, max_length=max_length)
        summary = generated[0] if generated else "Unable to generate summary"

        # Estimate confidence based on output length and coherence
        confidence = min(0.95, 0.5 + len(summary.split()) * 0.02)

        return {
            "summary": summary,
            "language": detected_lang,
            "confidence": confidence,
        }

    def generate(
        self,
        description: str,
        language: str = "python",
        max_length: int = 256,
        context: str | None = None,
    ) -> dict[str, Any]:
        """Generate code from natural language description.

        Args:
            description: What the code should do
            language: Target programming language
            max_length: Maximum code length
            context: Optional code context

        Returns:
            Dict with code, language, tokens_generated
        """
        if context:
            prompt = f"# Context:\n{context}\n\n# Generate {language} code to: {description}"
        else:
            prompt = f"Generate {language} code to: {description}"

        generated = self._generate(prompt, max_length=max_length)
        code = generated[0] if generated else ""

        return {
            "code": code,
            "language": language,
            "tokens_generated": len(self._tokenizer.encode(code)),
        }

    def translate(
        self,
        code: str,
        source_language: str,
        target_language: str,
        max_length: int = 512,
    ) -> dict[str, Any]:
        """Translate code between programming languages.

        Args:
            code: Source code to translate
            source_language: Source language
            target_language: Target language
            max_length: Maximum output length

        Returns:
            Dict with translated_code, source_language, target_language, confidence
        """
        prompt = f"Translate this {source_language} code to {target_language}:\n{code}"

        generated = self._generate(prompt, max_length=max_length)
        translated = generated[0] if generated else ""

        # Confidence based on output validity
        confidence = 0.7 if translated and len(translated) > 10 else 0.3

        return {
            "translated_code": translated,
            "source_language": source_language,
            "target_language": target_language,
            "confidence": confidence,
        }

    def complete(
        self,
        code_prefix: str,
        language: str | None = None,
        max_length: int = 128,
        num_suggestions: int = 3,
    ) -> dict[str, Any]:
        """Generate code completions.

        Args:
            code_prefix: Code to complete
            language: Programming language
            max_length: Maximum completion length
            num_suggestions: Number of suggestions to generate

        Returns:
            Dict with suggestions and language
        """
        detected_lang = language or self._detect_language(code_prefix)
        prompt = f"Complete this {detected_lang} code:\n{code_prefix}"

        generated = self._generate(
            prompt,
            max_length=max_length,
            num_beams=max(num_suggestions, 4),
            num_return_sequences=num_suggestions,
        )

        suggestions = []
        for i, completion in enumerate(generated):
            # Calculate confidence (higher for earlier beams)
            confidence = max(0.5, 0.95 - i * 0.1)
            suggestions.append({
                "completion": completion,
                "confidence": confidence,
            })

        return {
            "suggestions": suggestions,
            "language": detected_lang,
        }

    def understand(
        self,
        code: str,
        language: str | None = None,
    ) -> dict[str, Any]:
        """Analyze code semantics and structure.

        Args:
            code: Source code to analyze
            language: Programming language

        Returns:
            Dict with purpose, elements, data_flow, language
        """
        detected_lang = language or self._detect_language(code)

        # Generate purpose summary
        purpose_prompt = f"What does this {detected_lang} code do: {code}"
        purpose_generated = self._generate(purpose_prompt, max_length=100)
        purpose = purpose_generated[0] if purpose_generated else "Unable to determine purpose"

        # Extract code elements using regex patterns
        elements = []

        # Find functions/methods
        func_patterns = {
            "python": r"def\s+(\w+)\s*\(",
            "javascript": r"(?:function\s+(\w+)|(\w+)\s*=\s*(?:async\s+)?(?:function|\([^)]*\)\s*=>))",
            "java": r"(?:public|private|protected)?\s*(?:static\s+)?(?:\w+\s+)+(\w+)\s*\(",
        }
        pattern = func_patterns.get(detected_lang, r"(?:def|function|func)\s+(\w+)")
        for match in re.finditer(pattern, code):
            name = match.group(1) or (match.group(2) if match.lastindex > 1 else "unknown")
            if name:
                elements.append({
                    "name": name,
                    "element_type": "function",
                    "description": f"Function {name}",
                })

        # Find classes
        class_pattern = r"class\s+(\w+)"
        for match in re.finditer(class_pattern, code):
            elements.append({
                "name": match.group(1),
                "element_type": "class",
                "description": f"Class {match.group(1)}",
            })

        # Simple data flow analysis
        data_flow = []
        if "return" in code:
            data_flow.append("Returns a value")
        if "=" in code:
            data_flow.append("Contains variable assignments")
        if "for" in code or "while" in code:
            data_flow.append("Contains iteration/loops")
        if "if" in code:
            data_flow.append("Contains conditional logic")

        return {
            "purpose": purpose,
            "elements": elements,
            "data_flow": data_flow,
            "language": detected_lang,
        }

    def detect_defects(
        self,
        code: str,
        language: str | None = None,
    ) -> dict[str, Any]:
        """Identify potential bugs and issues.

        Args:
            code: Source code to analyze
            language: Programming language

        Returns:
            Dict with defects, code_quality_score, language
        """
        detected_lang = language or self._detect_language(code)
        defects = []

        # Pattern-based defect detection
        defect_patterns = [
            # Python specific
            (r"except\s*:", "medium", "Bare except clause catches all exceptions", "Specify exception type"),
            (r"eval\s*\(", "high", "Use of eval() is dangerous", "Avoid eval() or use ast.literal_eval()"),
            (r"exec\s*\(", "high", "Use of exec() is dangerous", "Avoid exec() for security"),
            (r"import\s+\*", "low", "Wildcard import may cause namespace pollution", "Import specific names"),

            # General patterns
            (r"TODO|FIXME|HACK|XXX", "low", "Contains TODO/FIXME comment", "Address or remove the TODO"),
            (r"password\s*=\s*['\"][^'\"]+['\"]", "high", "Hardcoded password detected", "Use environment variables"),
            (r"print\s*\(", "low", "Debug print statement", "Remove or convert to proper logging"),

            # Error handling
            (r"except.*:\s*pass\s*$", "medium", "Silent exception handling", "Log or handle the exception"),

            # Security
            (r"sql\s*=.*\+.*input|query.*\+.*user", "high", "Potential SQL injection", "Use parameterized queries"),
        ]

        for pattern, severity, description, suggestion in defect_patterns:
            if re.search(pattern, code, re.IGNORECASE | re.MULTILINE):
                match = re.search(pattern, code, re.IGNORECASE | re.MULTILINE)
                line_hint = None
                if match:
                    # Estimate line number
                    line_num = code[:match.start()].count("\n") + 1
                    line_hint = f"Around line {line_num}"

                defects.append({
                    "severity": severity,
                    "description": description,
                    "line_hint": line_hint,
                    "suggestion": suggestion,
                })

        # Calculate quality score (1.0 = perfect, 0.0 = terrible)
        high_count = sum(1 for d in defects if d["severity"] == "high")
        medium_count = sum(1 for d in defects if d["severity"] == "medium")
        low_count = sum(1 for d in defects if d["severity"] == "low")

        quality_score = max(0.0, 1.0 - (high_count * 0.3) - (medium_count * 0.15) - (low_count * 0.05))

        return {
            "defects": defects,
            "code_quality_score": round(quality_score, 2),
            "language": detected_lang,
        }

    def detect_clones(
        self,
        code: str,
        reference_snippets: list[str],
        threshold: float = 0.8,
    ) -> dict[str, Any]:
        """Find similar/duplicate code patterns.

        Uses CodeT5+ encoder embeddings to compute semantic similarity.

        Args:
            code: Source code to check
            reference_snippets: Code snippets to compare against
            threshold: Similarity threshold (0.0 to 1.0)

        Returns:
            Dict with matches and is_original flag
        """
        # Get embeddings for all code
        all_code = [code] + reference_snippets
        embeddings = self._get_embeddings(all_code)

        code_embedding = embeddings[0:1]
        ref_embeddings = embeddings[1:]

        # Compute cosine similarities
        similarities = torch.nn.functional.cosine_similarity(
            code_embedding.expand(ref_embeddings.size(0), -1),
            ref_embeddings,
        )

        matches = []
        for i, sim in enumerate(similarities.tolist()):
            if sim >= threshold:
                # Determine clone type based on similarity
                if sim >= 0.99:
                    clone_type = "exact"
                elif sim >= 0.9:
                    clone_type = "renamed"
                else:
                    clone_type = "structural"

                matches.append({
                    "reference_index": i,
                    "similarity": round(sim, 4),
                    "clone_type": clone_type,
                })

        # Sort by similarity descending
        matches.sort(key=lambda x: x["similarity"], reverse=True)

        return {
            "matches": matches,
            "is_original": len(matches) == 0,
        }
