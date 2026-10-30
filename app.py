
from pathlib import Path
import joblib
import pandas as pd
import logging
import sys
import time
import uuid
from flask import Flask, jsonify, request, g

app = Flask(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)

logger = logging.getLogger("student-result-api")
@app.before_request
def log_request_start():
    g.request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    g.start_time = time.perf_counter()


@app.after_request
def log_request_end(response):
    duration = (time.perf_counter() - g.start_time) * 1000

    log_method = logger.warning if response.status_code >= 400 else logger.info

    log_method(
        "event=request_completed request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
        g.request_id,
        request.method,
        request.path,
        response.status_code,
        duration
    )

    response.headers["X-Request-ID"] = g.request_id
    return response

MODEL_PATH = Path("student_result_model.pkl")

FEATURES = [
    "attendance",
    "internal_marks",
    "assignment_marks",
    "previous_score"
]


def load_model():
    if not MODEL_PATH.exists():
        logger.error("event=model_load_failed model_file=%s", MODEL_PATH)
        raise FileNotFoundError(
            "student_result_model.pkl was not found. Run the training pipeline first."
        )

    try:
        model = joblib.load(MODEL_PATH)
        logger.info("event=model_loaded model_file=%s", MODEL_PATH)
        return model
    except Exception:
        logger.exception("event=model_load_failed model_file=%s", MODEL_PATH)
        raise


@app.get("/")
def health_check():
    return jsonify({
        "status": "ok",
        "service": "student-result-prediction",
        "version": "2.0"
    })
@app.get("/health")
def health():
    return jsonify({
        "status": "healthy",
        "service": "student-result-prediction"
    }), 200

@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict) or not data:
        logger.warning(
            "event=prediction_validation_failed request_id=%s reason=invalid_json",
            g.request_id
        )
        return jsonify({"error": "JSON request body is required"}), 400

    missing_fields = [feature for feature in FEATURES if feature not in data]

    if missing_fields:
        logger.warning(
            "event=prediction_validation_failed request_id=%s missing_fields=%s",
            g.request_id,
            missing_fields
        )
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing_fields
        }), 400

    try:
        sample = pd.DataFrame([
            {feature: float(data[feature]) for feature in FEATURES}
        ])
    except (TypeError, ValueError):
        logger.warning(
            "event=prediction_validation_failed request_id=%s reason=non_numeric_values",
            g.request_id
        )
        return jsonify({"error": "Feature values must be numeric"}), 400

    try:
        model = load_model()
        prediction_code = int(model.predict(sample)[0])
        prediction = "PASS" if prediction_code == 1 else "FAIL"

        logger.info(
            "event=prediction_success request_id=%s prediction=%s",
            g.request_id,
            prediction
        )

        return jsonify({
            "prediction": prediction,
            "prediction_code": prediction_code
        })

    except Exception:
        logger.exception(
            "event=prediction_failed request_id=%s",
            g.request_id
        )
        return jsonify({"error": "Internal prediction error"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
