from flask import Flask, request, render_template
import pandas as pd
import pickle
df = pd.read_csv("tweet_emotions.csv")


data = 'model.pkl'
model = pickle.load(open(data, 'rb'))
vect = pickle.load(open('transform.pkl', 'rb'))
tfidf = pickle.load(open('transform1.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    comment = [x for x in request.form.values()]
    print(comment)

    x = df.content.values

    x = vect.transform(comment)
    x_tfidf = tfidf.transform(x)

    o = model.predict(x_tfidf)
    print(o)

    if o[0] == 0:
        return render_template('index.html', prog='Anger🤬')
    elif o[0] == 1:
        return render_template('index.html', prog='boredom😒')
    elif o[0] == 2:
        return render_template('index.html', prog='empty😑')
    elif o[0] == 3:
        return render_template('index.html', prog='enthusiasm😄')
    elif o[0] == 4:
        return render_template('index.html', prog='fun😆')
    elif o[0] == 5:
        return render_template('index.html', prog='happiness😁')
    elif o[0] == 6:
        return render_template('index.html', prog='hate😡')
    elif o[0] == 7:
        return render_template('index.html', prog='love💕')
    elif o[0] == 8:
        return render_template('index.html', prog='neutral😎')
    elif o[0] == 9:
        return render_template('index.html', prog='relief😌')
    elif o[0] == 10:
        return render_template('index.html', prog='sadness😔')
    elif o[0] == 11:
        return render_template('index.html', prog='surprise🥳')
    else:
        return render_template('index.html', prog='worry🤕')


if __name__ == '__main__':
    app.run(debug=True)

