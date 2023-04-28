from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

app = Flask(__name__, template_folder='templates')

model_path = 'app/models/model_20_15_1'
loaded_model = tf.keras.models.load_model(model_path)

def replace_outliers_with_median(data):
    median = np.median(data)
    # Определение Isolation Forest модели
    model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    # Обучение модели на данных
    model.fit(data)
    # Предсказание выбросов на данных
    outliers = model.predict(data)
    # Замена выбросов на медиану
    data[outliers == -1] = median
    return data

@app.route('/', methods=['POST', 'GET'])
def home():
    if request.method == 'POST':
        # Получение данных от пользователя
        density = float(request.form['Плотность, кг/м3'])
        modulus = float(request.form['Модуль упругости, ГПа'])
        hardener = float(request.form['Количество отвердителя, м.%'])
        epoxy = float(request.form['Содержание эпоксидных групп, %_2'])
        flash = float(request.form['Температура вспышки, С_2'])
        surface_density = float(request.form['Поверхностная плотность, г/м2'])
        resin_consumption = float(request.form['Потребление смолы, г/м2'])
        angle = float(request.form['Угол нашивки, град'])
        pitch = float(request.form['Шаг нашивки'])
        weft_density = float(request.form['Плотность нашивки'])

        # Создание массива из данных
        input_data = np.array([[density, modulus, hardener, epoxy, flash, surface_density,
                                resin_consumption, angle, pitch, weft_density]])
        
        # Замена выбросов на медиану
        input_data = replace_outliers_with_median(input_data)

        # Предсказание с помощью модели
        prediction = loaded_model(input_data)

        # Отображение результата на главной странице
        return render_template('index.html', prediction=prediction[0].numpy())
    else:
        return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Получение данных от пользователя
    density = float(request.form['Плотность, кг/м3'])
    modulus = float(request.form['Модуль упругости, ГПа'])
    hardener = float(request.form['Количество отвердителя, м.%'])
    epoxy = float(request.form['Содержание эпоксидных групп, %_2'])
    flash = float(request.form['Температура вспышки, С_2'])
    surface_density = float(request.form['Поверхностная плотность, г/м2'])
    resin_consumption = float(request.form['Потребление смолы, г/м2'])
    angle = float(request.form['Угол нашивки, град'])
    pitch = float(request.form['Шаг нашивки'])
    weft_density = float(request.form['Плотность нашивки'])

    # Создание массива из данных
    input_data = np.array([[density, modulus, hardener, epoxy, flash, surface_density,
                            resin_consumption, angle, pitch, weft_density]])
    
    # Замена выбросов на медиану
    input_data = replace_outliers_with_median(input_data)

     # Разделение выборки на x и y
    x = input_data
    y = None

    # Создание экземпляра StandardScaler и масштабирование данных в x
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    
    # Предсказание с помощью модели
    prediction = loaded_model(input_data)

    # Отображение результата на той же странице
    return render_template('index.html', prediction=prediction[0][0].numpy())

if __name__ == '__main__':
    app.run(debug=True)