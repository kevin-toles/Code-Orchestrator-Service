"""
Code-Orchestrator-Service - CodeT5+ API Endpoints

WBS: Architecture requirement for CodeT5+ integration
Exposes CodeT5+ capabilities as REST endpoints.

Capabilities:
- POST /v1/codet5/summarize - Generate natural language description of code
- POST /v1/codet5/generate - Generate code from natural language
- POST /v1/codet5/translate - Convert between programming languages
- POST /v1/codet5/complete - Predict next tokens in code
- POST /v1/codet5/understand - Semantic understanding of code structure
- POST /v1/codet5/detect-defects - Identify potential bugs in code
- POST /v1/codet5/detect-clones - Find similar/duplicate code patterns

Architecture Role: GENERATOR (Sous Chef)
Reference: KITCHEN_BRIGADE_ARCHITECTURE.md, ARCHITECTURE.md
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from src.core.logging import get_logger
from src.models.codet5_model import CodeT5Model, get_codet5_model

logger = get_logger(__name__)

# Router with /v1/codet5 prefix
router = APIRouter(prefix="/v1/codet5", tags=["codet5"])


# ============================================================================
# Request/Response Models
# ============================================================================


class Language(str, Enum):
    """Supported programming languages for translation."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    JAVA = "java"
    CPP = "cpp"
    CSHARP = "csharp"
    GO = "go"
    RUST = "rust"
    TYPESCRIPT = "typescript"
    RUBY = "ruby"
    PHP = "php"


class SummarizeRequest(BaseModel):
    """Request to summarize code."""
    code: str = Field(..., description="Source code to summarize", min_length=1)
    language: Language | None = Field(None, description="Programming language (auto-detected if not provided)")
    max_length: int = Field(128, ge=16, le=512, description="Maximum summary length in tokens")


class SummarizeResponse(BaseModel):
    """Response with code summary."""
    summary: str = Field(..., description="Natural language summary of the code")
    language: str = Field(..., description="Detected or specified language")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence score")


class GenerateRequest(BaseModel):
    """Request to generate code from natural language."""
    description: str = Field(..., description="Natural language description of code to generate", min_length=1)
    language: Language = Field(Language.PYTHON, description="Target programming language")
    max_length: int = Field(256, ge=32, le=1024, description="Maximum generation length in tokens")
    context: str | None = Field(None, description="Optional code context for better generation")


class GenerateResponse(BaseModel):
    """Response with generated code."""
    code: str = Field(..., description="Generated source code")
    language: str = Field(..., description="Target language")
    tokens_generated: int = Field(..., description="Number of tokens generated")


class TranslateRequest(BaseModel):
    """Request to translate code between languages."""
    code: str = Field(..., description="Source code to translate", min_length=1)
    source_language: Language = Field(..., description="Source programming language")
    target_language: Language = Field(..., description="Target programming language")
    max_length: int = Field(512, ge=32, le=2048, description="Maximum output length in tokens")


class TranslateResponse(BaseModel):
    """Response with translated code."""
    translated_code: str = Field(..., description="Translated source code")
    source_language: str = Field(..., description="Source language")
    target_language: str = Field(..., description="Target language")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Translation confidence")


class CompleteRequest(BaseModel):
    """Request to complete code."""
    code_prefix: str = Field(..., description="Code prefix to complete from", min_length=1)
    language: Language | None = Field(None, description="Programming language")
    max_length: int = Field(128, ge=16, le=512, description="Maximum completion length")
    num_suggestions: int = Field(3, ge=1, le=5, description="Number of completion suggestions")


class CompletionSuggestion(BaseModel):
    """A single completion suggestion."""
    completion: str = Field(..., description="Suggested completion")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")


class CompleteResponse(BaseModel):
    """Response with code completions."""
    suggestions: list[CompletionSuggestion] = Field(..., description="Completion suggestions")
    language: str = Field(..., description="Detected or specified language")


class UnderstandRequest(BaseModel):
    """Request to analyze code semantics."""
    code: str = Field(..., description="Source code to analyze", min_length=1)
    language: Language | None = Field(None, description="Programming language")


class CodeElement(BaseModel):
    """An identified code element."""
    name: str = Field(..., description="Element name")
    element_type: str = Field(..., description="Type (function, class, variable, etc.)")
    description: str = Field(..., description="What this element does")


class UnderstandResponse(BaseModel):
    """Response with code understanding."""
    purpose: str = Field(..., description="High-level purpose of the code")
    elements: list[CodeElement] = Field(..., description="Identified code elements")
    data_flow: list[str] = Field(..., description="Data flow description")
    language: str = Field(..., description="Detected or specified language")


class DefectRequest(BaseModel):
    """Request to detect defects in code."""
    code: str = Field(..., description="Source code to analyze", min_length=1)
    language: Language | None = Field(None, description="Programming language")


class Defect(BaseModel):
    """A detected defect."""
    severity: str = Field(..., description="Severity level (high, medium, low)")
    description: str = Field(..., description="Description of the defect")
    line_hint: str | None = Field(None, description="Approximate location hint")
    suggestion: str = Field(..., description="Suggested fix")


class DefectResponse(BaseModel):
    """Response with detected defects."""
    defects: list[Defect] = Field(..., description="Detected defects")
    code_quality_score: float = Field(..., ge=0.0, le=1.0, description="Overall code quality score")
    language: str = Field(..., description="Detected or specified language")


class CloneRequest(BaseModel):
    """Request to detect code clones."""
    code: str = Field(..., description="Source code to check for clones", min_length=1)
    reference_snippets: list[str] = Field(..., description="Code snippets to compare against", min_length=1)
    threshold: float = Field(0.8, ge=0.5, le=1.0, description="Similarity threshold")


class CloneMatch(BaseModel):
    """A detected clone match."""
    reference_index: int = Field(..., description="Index of matching reference snippet")
    similarity: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    clone_type: str = Field(..., description="Clone type (exact, renamed, structural)")


class CloneResponse(BaseModel):
    """Response with detected clones."""
    matches: list[CloneMatch] = Field(..., description="Detected clone matches")
    is_original: bool = Field(..., description="Whether code appears to be original")


# ============================================================================
# API Endpoints
# ============================================================================


@router.post("/summarize", response_model=SummarizeResponse)
async def summarize_code(request: SummarizeRequest) -> SummarizeResponse:
    """Generate natural language summary of code.

    Uses CodeT5+ encoder-decoder to understand code semantics and
    generate a human-readable description.

    Args:
        request: Code to summarize with optional language hint

    Returns:
        Natural language summary of what the code does
    """
    logger.info("codet5_summarize_request", code_length=len(request.code))

    try:
        model = get_codet5_model()
        result = model.summarize(
            code=request.code,
            language=request.language.value if request.language else None,
            max_length=request.max_length,
        )
        return SummarizeResponse(**result)

    except Exception as e:
        logger.error("codet5_summarize_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Summarization failed: {e}",
        )


@router.post("/generate", response_model=GenerateResponse)
async def generate_code(request: GenerateRequest) -> GenerateResponse:
    """Generate code from natural language description.

    Uses CodeT5+ decoder to generate source code from a
    natural language specification.

    Args:
        request: Description of code to generate

    Returns:
        Generated source code in the specified language
    """
    logger.info("codet5_generate_request", description_length=len(request.description))

    try:
        model = get_codet5_model()
        result = model.generate(
            description=request.description,
            language=request.language.value,
            max_length=request.max_length,
            context=request.context,
        )
        return GenerateResponse(**result)

    except Exception as e:
        logger.error("codet5_generate_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Code generation failed: {e}",
        )


@router.post("/translate", response_model=TranslateResponse)
async def translate_code(request: TranslateRequest) -> TranslateResponse:
    """Translate code between programming languages.

    Uses CodeT5+ encoder-decoder trained on NL↔Code pairs
    to perform cross-language translation.

    Args:
        request: Code to translate with source and target languages

    Returns:
        Translated code in the target language
    """
    logger.info(
        "codet5_translate_request",
        source=request.source_language.value,
        target=request.target_language.value,
    )

    try:
        model = get_codet5_model()
        result = model.translate(
            code=request.code,
            source_language=request.source_language.value,
            target_language=request.target_language.value,
            max_length=request.max_length,
        )
        return TranslateResponse(**result)

    except Exception as e:
        logger.error("codet5_translate_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Translation failed: {e}",
        )


@router.post("/complete", response_model=CompleteResponse)
async def complete_code(request: CompleteRequest) -> CompleteResponse:
    """Predict next tokens to complete code.

    Uses CodeT5+ to generate likely continuations of the
    provided code prefix.

    Args:
        request: Code prefix to complete

    Returns:
        Multiple completion suggestions ranked by confidence
    """
    logger.info("codet5_complete_request", prefix_length=len(request.code_prefix))

    try:
        model = get_codet5_model()
        result = model.complete(
            code_prefix=request.code_prefix,
            language=request.language.value if request.language else None,
            max_length=request.max_length,
            num_suggestions=request.num_suggestions,
        )
        return CompleteResponse(**result)

    except Exception as e:
        logger.error("codet5_complete_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Code completion failed: {e}",
        )


@router.post("/understand", response_model=UnderstandResponse)
async def understand_code(request: UnderstandRequest) -> UnderstandResponse:
    """Analyze code semantics and structure.

    Uses CodeT5+ to understand what code does, identify key elements,
    and describe data flow.

    Args:
        request: Code to analyze

    Returns:
        Semantic understanding including purpose, elements, and data flow
    """
    logger.info("codet5_understand_request", code_length=len(request.code))

    try:
        model = get_codet5_model()
        result = model.understand(
            code=request.code,
            language=request.language.value if request.language else None,
        )
        return UnderstandResponse(**result)

    except Exception as e:
        logger.error("codet5_understand_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Code understanding failed: {e}",
        )


@router.post("/detect-defects", response_model=DefectResponse)
async def detect_defects(request: DefectRequest) -> DefectResponse:
    """Identify potential bugs and issues in code.

    Uses CodeT5+ to analyze code for common defects, anti-patterns,
    and potential bugs.

    Args:
        request: Code to analyze for defects

    Returns:
        List of detected defects with severity and suggested fixes
    """
    logger.info("codet5_detect_defects_request", code_length=len(request.code))

    try:
        model = get_codet5_model()
        result = model.detect_defects(
            code=request.code,
            language=request.language.value if request.language else None,
        )
        return DefectResponse(**result)

    except Exception as e:
        logger.error("codet5_detect_defects_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Defect detection failed: {e}",
        )


@router.post("/detect-clones", response_model=CloneResponse)
async def detect_clones(request: CloneRequest) -> CloneResponse:
    """Find similar/duplicate code patterns.

    Uses CodeT5+ embeddings to detect code clones by comparing
    semantic similarity between code snippets.

    Args:
        request: Code to check with reference snippets to compare against

    Returns:
        List of clone matches above the similarity threshold
    """
    logger.info(
        "codet5_detect_clones_request",
        code_length=len(request.code),
        reference_count=len(request.reference_snippets),
    )

    try:
        model = get_codet5_model()
        result = model.detect_clones(
            code=request.code,
            reference_snippets=request.reference_snippets,
            threshold=request.threshold,
        )
        return CloneResponse(**result)

    except Exception as e:
        logger.error("codet5_detect_clones_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Clone detection failed: {e}",
        )


@router.get("/health")
async def codet5_health() -> dict[str, Any]:
    """Check CodeT5+ model health and status."""
    try:
        model = get_codet5_model()
        return {
            "status": "healthy",
            "model": "Salesforce/codet5p-220m",
            "device": model.device,
            "loaded": model.is_loaded,
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
        }
