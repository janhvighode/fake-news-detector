from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/detector")
def detector():
    return render_template("detector.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/predict", methods=["POST"])
def predict():

    news = request.form["news"]

    vector = vectorizer.transform([news])

    prediction = model.predict(vector)
    probability = model.predict_proba(vector)

    trust_score = round(probability.max()*100,2)

    if prediction[0] == 1:
        result = "Real News"
    else:
        result = "Fake News"

    return render_template(
        "detector.html",
        prediction_text=result,
        trust_score=str(trust_score)+"%"
    )

if __name__ == "__main__":
    app.run(debug=True)
