from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from utils import prepare_features_from_raw

app = Flask(__name__)
CORS(app)

# Load models
MODELS = {
    "lr": joblib.load("models/lr_model.joblib"),
    "rf": joblib.load("models/rf_model.joblib"),
}


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "House Price Prediction API",
        "endpoints": {
            "POST /predict?model=lr|rf": {
                "expects_json": {
                    "Size_sqft": "number",
                    "Bedrooms": "int",
                    "Bathrooms": "int",
                    "YearBuilt": "int",
                    "Location": "City|Suburb|Rural"
                }
            }
        }
    })


@app.route("/predict", methods=["POST"])
def predict():
    # choose model
    choice = (request.args.get("model")).lower()
    if choice not in MODELS:
        return jsonify({"error": "Unknown model. Use model=lr or model=rf"}), 400
    model = MODELS[choice]

    # read payload
    data = request.get_json(silent=True) or {}
    required = ["Size_sqft", "Bedrooms", "Bathrooms", "YearBuilt", "Location"]
    missing = [k for k in required if k not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        x_new = prepare_features_from_raw(data)
        pred = float(model.predict(x_new)[0])
    except Exception as e:
        return jsonify({"error": f"Failed to prepare/predict: {e}"}), 500

    return jsonify({
        "model": "linear_regression" if choice == "lr" else "random_forest",
        "input": {
            "Size_sqft": float(data["Size_sqft"]),
            "Bedrooms": int(data["Bedrooms"]),
            "Bathrooms": int(data["Bathrooms"]),
            "YearBuilt": int(data["YearBuilt"]),
            "Location": str(data["Location"])
        },
        "prediction": round(pred, 2)
    })


app.run(host="0.0.0.0", port=8080, debug=False)
