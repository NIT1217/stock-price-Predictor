from flask import Flask, request, render_template
import pickle
import ast
import numpy as np

app = Flask(__name__)

# Load model once
with open("model.pkl", "rb") as f:
    model = pickle.load(f)


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", response=None)


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("lastdata")

    content = file.read().decode("utf-8")

    values = ast.literal_eval(content)        # list
    values = np.array(values, dtype=float)   # numpy

    values = values.reshape(1, -1)

    prediction = model.predict(values)

    result = float(prediction.flatten()[0])
     

    return render_template("index.html", response=result)


if __name__ == "__main__":
    app.run(debug=True)


    
    


