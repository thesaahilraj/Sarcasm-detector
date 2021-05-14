from flask import Flask, render_template, request
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

app = Flask(__name__, template_folder='template')

@app.route('/')
def index():
  return render_template('input.html')

@app.route('/predict', methods=['POST'])
def predict():
  
  Filename = "sarcasm-detector.pkl"
  with open(Filename, 'rb') as file:  
    model = pickle.load(file)
  
  text = request.form['text']
  textlen = len(text)

  onehot=[one_hot(text, textlen)]
  result = pad_sequences(onehot, padding='pre', maxlen=textlen)
  
  prediction = model.predict(result)
  if prediction > 0.5:
    ans = "Sarcastic"
  else:
    ans = "Not Sarcastic"
  return render_template('output.html', text = text, prediction = prediction, ans = ans)


if __name__ == "__main__":
  app.run(debug=True)