from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def index():
    df = pd.read_csv("data.csv")
    df.isnull().sum()
    data_r = df.fillna(method='ffill')
    data_r
    k = list(data_r.columns)
    X = data_r[k]
    X = df['Year']
    Y = df['Jan']
    train_X = X[:102].reshape(-1, 1)
    train_Y = Y[:102].reshape(-1, 1)
    test_X = X[-14:].reshape(-1, 1)
    test_Y = Y[-14:].reshape(-1, 1)
    df = pd.read_csv('data-cp.csv')
    autm = df[df.Season == 'Kharif']
    py = pd.DataFrame(autm.Crop_Year)
    reg = LinearRegression()
    reg.fit(train_X, train_Y)
    test_Y_pred = reg.predict(test_X)
    train_Y_pred = reg.predict(train_X)
    score_X = reg.score(train_X, train_Y)
    new_pred = np.array([[2018], [2019], [2020], [2021], [2022], [2023], [2024], [
        2024], [2024], [2025], [2026], [2027], [2028], [2029], [2030]])
    train_Y_pred = reg.predict(py)
    print(train_Y_pred)

    if request.method == "POST":
        x_atr = request.form['x-axis']
        y_atr = request.form['y-axis']
        return redirect(url_for('plots', x=x_atr, y=y_atr))

    return render_template('index.html', train_Y_pred=train_Y_pred)
    app = Flask(__name__)


if __name__ == "__main__":
    app.run(debug=True)
