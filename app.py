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
  try:
    model = pickle.load(open('sarcasm-detector.pkl','rb'))
  except:
    return 'Load Model Error'
  
  text = request.form['text']
  textlen = len(text)

  try:
    onehot=[one_hot(text, textlen)]
    result = pad_sequences(onehot, padding='pre', maxlen=textlen)
  except:
    return 'Input Preprocessing Error'

  prediction = model.predict(result)
  if prediction > 0.5:
    ans = "Sarcastic"
  else:
    ans = "Not Sarcastic"
  return render_template('output.html', text = text, prediction = prediction, ans = ans)


if __name__ == "__main__":
  app.run(debug=True)