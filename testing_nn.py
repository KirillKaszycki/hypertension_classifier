from keras import models

# Мы сохранили обученную нейросеть в файл. Сейчас мы её распаковываем для использования в боте

if __name__ == '__main__':
    model = models.load_model('hypertension_classifier_neural_network.h5')

    # Тут делаем тестовый предикт
    # Сообщение в телеграме должно сохраняться в двумерный массив
    data_pred = [[50, 1, 3, 120, 233, 0, 150, 1, 2.3, 1, 2]]
    prediction = model.predict(data_pred)
    print(prediction)
