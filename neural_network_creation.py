import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os


class NeuralNetwork:
    def __init__(self):
        self.model = None

    def load_data(self, file_path):
        file_path = 'hypertension_data.csv'
        data = pd.read_csv(file_path)
        data = data.fillna(data.mean())
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data.drop(['ca', 'restecg', 'target'],
                                                                                          axis=1),
                                                    data['target'], test_size=0.3, random_state=42)

    def define_model(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(128, input_shape=(11,), activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

    def compile_model(self):
        self.model.compile(optimizer='adam',
                           loss='binary_crossentropy',
                           metrics=[tf.keras.metrics.Recall()])

    def train_model(self, epochs=100):
        self.model.fit(self.X_train, self.y_train, epochs=epochs)

    # Определяем веса модели и упаковываем её в файл <.h5>

    def evaluate_model(self):
        test_loss, test_recall = self.model.evaluate(self.X_test, self.y_test)
        print('Test recall:', test_recall)
        self.model.save('hypertension_classifier_neural_network.h5')

    # Метод для предсказания

    def predict(self, new_data):
        return self.model.predict(new_data)


# Вызываем все методы

if __name__ == '__main__':
    nn = NeuralNetwork()
    nn.load_data('hypertension_data.csv')
    nn.define_model()
    nn.compile_model()
    nn.train_model(epochs=100)
    nn.evaluate_model()


