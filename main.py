import logging
import time
import uuid
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

import rag
from guardrails import validate_input, validate_output, mask_pii
from monitoring import (
    record_request,
    get_metrics,
    get_request_history,
    get_latency_trend,
    get_daily_metrics,
)


# =====================================================
# 1. PROJECT CONFIGURATION
# =====================================================

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "artifacts" / "intent_classifier.joblib"

LOG_DIR = PROJECT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = (LOG_DIR / "app.log").resolve()


# =====================================================
# 2. LOGGING CONFIGURATION
# =====================================================

logger = logging.getLogger("hdfc_assistant")
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    file_handler = logging.FileHandler(
        LOG_FILE, mode="a", encoding="utf-8"
    )
    console_handler = logging.StreamHandler()

    log_format = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    file_handler.setFormatter(log_format)
    console_handler.setFormatter(log_format)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

print(f"LOG FILE PATH: {LOG_FILE}", flush=True)
logger.info("HDFC Bank AI Assistant logging initialized.")


# =====================================================
# 3. LOAD INTENT CLASSIFIER
# =====================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Intent classifier not found at {MODEL_PATH}. "
        "Confirm artifacts/intent_classifier.joblib exists."
    )

intent_classifier = joblib.load(MODEL_PATH)
logger.info("Intent classifier loaded successfully.")


# =====================================================
# 4. FASTAPI APPLICATION
# =====================================================

app = FastAPI(
    title="HDFC Custom LLM Pipeline API",
    version="1.1.0",
    description=(
        "Employee-facing HDFC Bank AI Assistant API "
        "with persistent monitoring."
    ),
)


# =====================================================
# 5. REQUEST SCHEMA
# =====================================================

class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=8000)


# =====================================================
# 6. REQUEST ID MIDDLEWARE
# =====================================================

@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Attach a unique ID to each HTTP request and log its latency."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    start_time = time.perf_counter()

    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    except Exception:
        logger.exception(
            "Unhandled HTTP request error | Request ID: %s | Path: %s",
            request_id,
            request.url.path,
        )
        raise

    finally:
        elapsed = time.perf_counter() - start_time

        logger.info(
            "HTTP request completed | Request ID: %s | "
            "Method: %s | Path: %s | Latency: %.3f seconds",
            request_id,
            request.method,
            request.url.path,
            elapsed,
        )


# =====================================================
# 7. ROOT ENDPOINT
# =====================================================

@app.get("/")
def home(request: Request):
    request_id = request.state.request_id

    logger.info(
        "Home endpoint accessed | Request ID: %s",
        request_id,
    )

    return {
        "status": "success",
        "message": "HDFC Bank AI Assistant API is running.",
        "request_id": request_id,
    }


# =====================================================
# 8. HEALTH ENDPOINT
# =====================================================

@app.get("/health")
def health(request: Request):
    request_id = request.state.request_id

    logger.info(
        "Health endpoint accessed | Request ID: %s",
        request_id,
    )

    return {
        "status": "healthy",
        "faq_records": len(rag.faq_df),
        "intent_classifier_loaded": intent_classifier is not None,
        "monitoring_storage": "sqlite",
        "request_id": request_id,
    }


# =====================================================
# 9. METRICS ENDPOINTS
# =====================================================

@app.get("/metrics")
def metrics(request: Request):
    """Return persistent, all-time request metrics."""
    request_id = request.state.request_id

    logger.info(
        "Metrics endpoint accessed | Request ID: %s",
        request_id,
    )

    result = get_metrics()
    result["request_id"] = request_id

    return result


@app.get("/metrics/history")
def metrics_history(
    request: Request,
    limit: int = 100,
    offset: int = 0,
):
    """Return recent request metadata without question or answer text."""
    request_id = request.state.request_id

    return {
        "request_id": request_id,
        "items": get_request_history(
            limit=limit,
            offset=offset,
        ),
    }


@app.get("/metrics/latency-trend")
def metrics_latency_trend(
    request: Request,
    limit: int = 100,
):
    """Return recent latency observations for charts."""
    request_id = request.state.request_id

    return {
        "request_id": request_id,
        "items": get_latency_trend(limit=limit),
    }


@app.get("/metrics/daily")
def metrics_daily(
    request: Request,
    days: int = 30,
):
    """Return daily request counts and average latency."""
    request_id = request.state.request_id

    return {
        "request_id": request_id,
        "items": get_daily_metrics(days=days),
    }


# =====================================================
# 10. ASK ENDPOINT
# =====================================================

@app.post("/ask")
def ask(payload: QuestionRequest, request: Request):
    """
    Validate the question, retrieve FAQs, generate an answer,
    and persist request metrics.
    """

    start_time = time.perf_counter()
    status = "failure"
    error_category = None
    request_id = request.state.request_id

    logger.info(
        "Ask request received | Request ID: %s",
        request_id,
    )

    try:
        # -------------------------------------------------
        # STEP 1: Validate question
        # -------------------------------------------------

        question = payload.question.strip()

        if not question:
            error_category = "invalid_question"

            raise HTTPException(
                status_code=422,
                detail="Please enter a valid question.",
            )

        # -------------------------------------------------
        # STEP 2: Input guardrails
        # -------------------------------------------------

        try:
            check = validate_input(question)

        except Exception:
            error_category = "input_guardrails_error"

            logger.exception(
                "Input guardrails failed | Request ID: %s",
                request_id,
            )

            raise HTTPException(
                status_code=500,
                detail="Input security validation failed.",
            )

        if not check.allowed:
            status = "blocked"
            error_category = "input_blocked"

            logger.warning(
                "Request blocked by input guardrails | Request ID: %s",
                request_id,
            )

            return {
                "status": "blocked",
                "answer": "Your request was blocked by security checks.",
                "sources": [],
                "request_id": request_id,
            }

        safe_question = check.text

        # -------------------------------------------------
        # STEP 3: Intent prediction
        # -------------------------------------------------

        try:
            intent = str(
                intent_classifier.predict([safe_question])[0]
            )

            logger.info(
                "Intent prediction completed | Request ID: %s",
                request_id,
            )

        except Exception:
            error_category = "intent_prediction"

            logger.exception(
                "Intent prediction failed | Request ID: %s",
                request_id,
            )

            raise HTTPException(
                status_code=500,
                detail="Intent prediction failed.",
            )

        # -------------------------------------------------
        # STEP 4: FAQ retrieval
        # -------------------------------------------------

        try:
            faqs = rag.retrieve_faqs(safe_question)

            logger.info(
                "FAQ retrieval completed | Count: %d | Request ID: %s",
                len(faqs),
                request_id,
            )

        except Exception:
            error_category = "faq_retrieval"

            logger.exception(
                "FAQ retrieval failed | Request ID: %s",
                request_id,
            )

            raise HTTPException(
                status_code=500,
                detail="FAQ retrieval failed.",
            )

        # -------------------------------------------------
        # STEP 5: Mask PII in retrieved context
        # -------------------------------------------------

        try:
            safe_faqs = [
                {
                    "question": mask_pii(faq["question"]),
                    "answer": mask_pii(faq["answer"]),
                    "class": faq["class"],
                    "similarity": faq["similarity"],
                }
                for faq in faqs
            ]

        except Exception:
            error_category = "context_pii_masking"

            logger.exception(
                "FAQ context preparation failed | Request ID: %s",
                request_id,
            )

            raise HTTPException(
                status_code=500,
                detail="Retrieved context could not be prepared safely.",
            )

        # -------------------------------------------------
        # STEP 6: Generate answer
        # -------------------------------------------------

        try:
            answer = rag.generate_answer(
                safe_question,
                safe_faqs,
            )

            logger.info(
                "Answer generation completed | Request ID: %s",
                request_id,
            )

        except Exception:
            error_category = "answer_generation"

            logger.exception(
                "Answer generation failed | Request ID: %s",
                request_id,
            )

            raise HTTPException(
                status_code=500,
                detail="Answer generation failed.",
            )

        # -------------------------------------------------
        # STEP 7: Validate generated output
        # -------------------------------------------------

        try:
            output_check = validate_output(answer)

        except Exception:
            error_category = "output_guardrails_error"

            logger.exception(
                "Output guardrails failed | Request ID: %s",
                request_id,
            )

            raise HTTPException(
                status_code=500,
                detail="Output security validation failed.",
            )

        if not output_check.allowed:
            status = "blocked"
            error_category = "output_blocked"

            logger.warning(
                "Generated response blocked by output guardrails "
                "| Request ID: %s",
                request_id,
            )

            return {
                "status": "blocked",
                "intent": intent,
                "answer": (
                    "The generated response did not pass security checks."
                ),
                "sources": [],
                "request_id": request_id,
            }

        # -------------------------------------------------
        # STEP 8: Prepare source evidence
        # -------------------------------------------------

        sources = [
            {
                "question": faq["question"],
                "class": faq["class"],
                "similarity": faq["similarity"],
            }
            for faq in safe_faqs
        ]

        # -------------------------------------------------
        # STEP 9: Return successful response
        # -------------------------------------------------

        status = "success"
        error_category = None

        return {
            "status": "success",
            "intent": intent,
            "answer": output_check.text,
            "sources": sources,
            "request_id": request_id,
        }

    except HTTPException as exc:
        if error_category is None:
            error_category = f"http_{exc.status_code}"

        raise

    except Exception:
        error_category = error_category or "unexpected_error"

        logger.exception(
            "Unexpected error while processing request "
            "| Request ID: %s",
            request_id,
        )

        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred.",
        )

    finally:
        # -------------------------------------------------
        # STEP 10: Persist monitoring metrics
        # -------------------------------------------------

        elapsed_seconds = time.perf_counter() - start_time

        try:
            record_request(
                status=status,
                latency_seconds=elapsed_seconds,
                error_category=error_category,
                endpoint="/ask",
                request_id=request_id,
            )

        except Exception:
            # Monitoring failure must not replace the actual API response.
            logger.exception(
                "Could not persist monitoring record | Request ID: %s",
                request_id,
            )

        logger.info(
            "Ask request finished | Request ID: %s | Status: %s | "
            "Error category: %s | Latency: %.3f seconds",
            request_id,
            status,
            error_category or "none",
            elapsed_seconds,
        )