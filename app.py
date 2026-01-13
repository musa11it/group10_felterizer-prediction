from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import numpy as np

# ✅ 1. Create Flask app FIRST
app = Flask(__name__)
CORS(app)

# ✅ 2. Load model
with open("fertilizer_model.pkl", "rb") as f:
    model = pickle.load(f)

# ✅ 3. Load label encoder
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# ✅ 4. Custom mapping
custom_names = {
    "Urea": "Germain",
    "NPK": "Benjamin",
    "DAP": "DAP Custom",
    "General Purpose": "General Custom",
    "Moisture Retaining": "Moisture Custom"
}

# ✅ 5. Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    features = [
        data["Moisture"],
        data["Temperature"],
        data["EC"],
        data["pH"],
        data["N"],
        data["P"],
        data["K"]
    ]

    features = np.array(features).reshape(1, -1)

    prediction = model.predict(features)[0]
    original_label = label_encoder.inverse_transform([prediction])[0]

    display_label = custom_names.get(original_label, original_label)

    return jsonify({"prediction": display_label})

# ✅ 6. Run app
if __name__ == "__main__":
    app.run(debug=True)
