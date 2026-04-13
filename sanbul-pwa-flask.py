import tensorflow as tf
from tensorflow import keras
print("TensorFlow version:", tf.__version__)

import numpy as np
import pandas as pd
from flask import Flask, render_template, request

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedShuffleSplit

np.random.seed(42)

# ── 데이터 로드 및 전처리 파이프라인 재구성 ──────────────────────
fires_raw = pd.read_csv("./sanbul2district-divby100.csv", sep=",")
fires_raw["burned_area"] = np.log(fires_raw["burned_area"] + 1)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(fires_raw, fires_raw["month"]):
    strat_train_set = fires_raw.loc[train_index]

fires_tr = strat_train_set.drop(["burned_area"], axis=1)

num_attribs = ['longitude', 'latitude', 'avg_temp', 'max_temp', 'max_wind_speed', 'avg_wind']
cat_attribs = ['month', 'day']

num_pipeline = Pipeline([('std_scaler', StandardScaler())])
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])
full_pipeline.fit(fires_tr)

# ── 저장된 모델 로드 ───────────────────────────────────────────
model = keras.models.load_model('fires_model.h5')

# ── Flask 앱 설정 ──────────────────────────────────────────────
app = Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/prediction', methods=['GET', 'POST'])
def lab():
    if request.method == 'POST':
        longitude      = float(request.form['longitude'])
        latitude       = float(request.form['latitude'])
        month          = request.form['month']
        day            = request.form['day']
        avg_temp       = float(request.form['avg_temp'])
        max_temp       = float(request.form['max_temp'])
        max_wind_speed = float(request.form['max_wind_speed'])
        avg_wind       = float(request.form['avg_wind'])

        input_data = pd.DataFrame([{
            'longitude'     : longitude,
            'latitude'      : latitude,
            'month'         : month,
            'day'           : day,
            'avg_temp'      : avg_temp,
            'max_temp'      : max_temp,
            'max_wind_speed': max_wind_speed,
            'avg_wind'      : avg_wind
        }])

        input_prepared = full_pipeline.transform(input_data)
        pred_log  = model.predict(input_prepared)[0][0]
        pred_area = max(0, round(float(np.expm1(pred_log)), 2))

        return render_template('result.html', prediction=pred_area)

    return render_template('prediction.html')

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)