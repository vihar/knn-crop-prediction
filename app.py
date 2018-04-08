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
    df = pd.read_csv("data.csv")
    attributes = df[:0]

    if request.method == 'POST':
        year = request.form['year-axis']
        month = request.form['month-axis']
        return redirect(url_for('prediction', year=year, month=month))
    return render_template('year_form.html', att=attributes)


@app.route('/prediction', methods=['GET'])
def prediction():

    df = pd.read_csv("data.csv")
    dfc = pd.read_csv("comparetemp.csv")
    dftr = pd.read_csv("temprang.csv")
    dfevo = pd.read_csv("evotran.csv")
    dfpre = pd.read_csv("precp.csv")
    dfwet = pd.read_csv("wetd.csv")

    year = request.args.get('year')
    month = request.args.get('month')

    # Data Cleaning
    df.isnull().sum()
    data_r = df.fillna(method='ffill')
    data_r
    k = list(data_r.columns)
    X = data_r[k]

    # Training Model
    X = df['Year']
    Y = df[month]

    K = dftr['Year']
    N = dftr[month]

    A = dfevo['Year']
    B = dfevo[month]

    C = dfpre['Year']
    D = dfpre[month]

    E = dfwet['Year']
    F = dfwet[month]

    train_X = X[:102].reshape(-1, 1)
    train_Y = Y[:102].reshape(-1, 1)

    train_K = K[:102].reshape(-1, 1)
    train_N = N[:102].reshape(-1, 1)

    train_A = A[:102].reshape(-1, 1)
    train_B = B[:102].reshape(-1, 1)

    train_C = C[:102].reshape(-1, 1)
    train_D = D[:102].reshape(-1, 1)

    train_E = E[:102].reshape(-1, 1)
    train_F = F[:102].reshape(-1, 1)

    # Model Definition and Fitting

    # Average Temparature
    reg = LinearRegression()
    reg.fit(train_X, train_Y)

    # Temperature Range
    reg_tr = LinearRegression()
    reg_tr.fit(train_K, train_N)

    # Evapo Transpiration
    reg_evo = LinearRegression()
    reg_evo.fit(train_A, train_B)

    # Precipitaion
    reg_pre = LinearRegression()
    reg_pre.fit(train_C, train_D)

    # Wet Day
    reg_wet = LinearRegression()
    reg_wet.fit(train_E, train_F)

    # Input List
    pred_list = []
    pred_list.append(int(year))
    single_x = np.array(pred_list)

    year_avg_temp = reg.predict(single_x)
    year_temp_range = reg_tr.predict(single_x)
    year_evo_tran = reg_evo.predict(single_x)
    year_precp = reg_pre.predict(single_x)
    year_wet = reg_wet.predict(single_x)

    hel = []
    for i in range(int(year), int(year) + 5):
        hel.append(int(i))

    arr = np.array(hel)
    multi_pred = arr.reshape(-1, 1)

    year_avg_temp_mul = reg.predict(multi_pred)
    year_temp_range_mul = reg_tr.predict(multi_pred)
    year_evo_tran_mul = reg_evo.predict(multi_pred)
    year_precp_mul = reg_pre.predict(multi_pred)
    year_wet_mul = reg_wet.predict(multi_pred)

    compare_dic = dict(zip(dfc['crop'], dfc['temperature']))

    for i in compare_dic:
        compare_dic[i] = abs(int(year_avg_temp) - compare_dic[i])

    euclid_dict = compare_dic
    import operator
    sorted_x = sorted(euclid_dict.items(), key=operator.itemgetter(1))

    prediction_crops = sorted_x[:3]

    return render_template('show.html', year=year,
                           month=month,
                           compare_dic=compare_dic,
                           sorted_x=sorted_x,
                           prediction_crops=prediction_crops,
                           year_avg_temp=year_avg_temp,
                           year_temp_range=year_temp_range,
                           year_evo_tran=year_evo_tran,
                           year_precp=year_precp,
                           year_wet=year_wet,
                           year_avg_temp_mul=year_avg_temp_mul,
                           year_temp_range_mul=year_temp_range_mul,
                           year_evo_tran_mul=year_evo_tran_mul,
                           year_precp_mul=year_precp_mul,
                           year_wet_mul=year_wet_mul
                           )


if __name__ == '__main__':
    app.run(debug=True)
