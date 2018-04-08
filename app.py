from flask import Flask, render_template, url_for, request, redirect
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST', 'GET'])
def input_form():
    if request.method == 'POST':
        year = request.form['year-axis']
        return redirect(url_for('prediction', year=year))
    return render_template('year_form.html')


@app.route('/prediction', methods=['GET'])
def prediction():
    df = pd.read_csv("data.csv")
    dfc = pd.read_csv("comparetemp.csv")
    year = request.args.get('year')

    # Data Cleaning
    df.isnull().sum()
    data_r = df.fillna(method='ffill')
    data_r
    k = list(data_r.columns)
    X = data_r[k]

    # Training Model
    X = df['Year']
    Y = df['Mar']
    # pd.Dataframe(Y,Z,k).mean()
    train_X = X[:102].reshape(-1, 1)
    train_Y = Y[:102].reshape(-1, 1)

    # Model Definition
    reg = LinearRegression()
    pred_list = []
    hel = []
    for i in range(int(year), int(year) + 5):
        print(i)
        hel.append(int(i))
    arr = np.array(hel)
    print(arr)
    multi_pred = arr.reshape(-1, 1)

    pred_list.append(int(year))
    print(pred_list)
    # Season Selection
    df = pd.read_csv('data-cp.csv')
    reg.fit(train_X, train_Y)
    single_x = np.array(pred_list)
    single_y = reg.predict(single_x)
    multi_y = reg.predict(multi_pred)

    compare_dic = dict(zip(dfc['crop'], dfc['temperature']))

    for i in compare_dic:
        compare_dic[i] = abs(int(single_y) - compare_dic[i])

    euclid_dict = compare_dic
    import operator

    sorted_x = sorted(euclid_dict.items(), key=operator.itemgetter(1))

    prediction_crops = sorted_x[:3]

    return render_template('show.html', year=year, single_y=single_y,
                           compare_dic=compare_dic, sorted_x=sorted_x, prediction_crops=prediction_crops,
                           multi_y = multi_y)


if __name__ == '__main__':
    app.run(debug=True)
