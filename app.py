from flask import Flask, render_template, request
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow import keras
model = keras.models.load_model('sarcasm-detector.h5')

app = Flask(__name__, template_folder='template')


@app.route('/')
def index():
    return render_template('input.html')


@app.route('/predict', methods=['POST'])
def predict():

    text = request.form['text']
    textlen = len(text)

    onehot = [one_hot(text, textlen)]
    result = pad_sequences(onehot, padding='pre', maxlen=textlen)

    prediction = model.predict(result)
    prediction = prediction[0]
    
    if prediction > 0.5:
        ans = "Sarcastic"
    else:
        ans = "Not Sarcastic"
    
    return render_template('output.html', text=text, prediction=round(prediction[0]*100), ans=ans)


if __name__ == "__main__":
    app.run(debug=True)
