from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)

# Load model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    url = data.get("url", "")

    X = vectorizer.transform([url])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0].max() * 100

    return jsonify({
        "prediction": "PHISHING ⚠️" if pred == 1 else "SAFE ✅",
        "confidence": round(prob, 2)
    })

if __name__ == "__main__":
    app.run(debug=True)
app.run(host="0.0.0.0", port=5000, debug=True)
