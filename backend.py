from flask import Flask, request, jsonify, render_template
import joblib
import re
import string

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("lr.model.jb")
vectorizer = joblib.load("vectorizer.jb")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', ' ', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text.strip()

@app.route("/verify", methods=["POST"])
def verify_claim():
    try:
        data = request.get_json()
        claim = data.get("claim", "")
        additional_info = data.get("additional_info", "")

        if not claim:
            return jsonify({"error": "No claim provided."})

        text = f"{claim} {additional_info}"
        cleaned = clean_text(text)
        vec = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]
        result = "Fake" if prediction == 0 else "Real"

        return jsonify({"result": result})
    except Exception as e:
        print("VERIFY ERROR:", e)
        return jsonify({"error": str(e)})

# Serve frontend HTML
@app.route("/")
def home():
    return render_template("verify.html")  # serve your frontend file

if __name__ == "__main__":
    app.run(debug=True)
