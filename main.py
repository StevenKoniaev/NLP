from flask import Flask, render_template, request, jsonify
#from flask_cors import CORS

from chat import get_responses

app = Flask(__name__)
# CORS(app)

@app.get("/")
def index_get():
    return render_template("base.html")

@app.post("/predict")
def predict():
    text = request.get_json().get("message")

    reponse = get_responses(text)
    message = {"answer" : reponse}

    return jsonify(message)


if __name__ == "__main__":
    app.run(debug=True)
