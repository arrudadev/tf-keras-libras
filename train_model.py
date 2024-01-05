import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Carregar os dados
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.array(data_dict['data'])
labels = np.array(data_dict['labels'])

# Dividir os dados em conjuntos de treino e teste
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Definir a arquitetura da rede neural
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(21, 2)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    # tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(np.max(labels) + 1, activation='softmax')
])

# Compilar o modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Treinar o modelo
model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

# Avaliar o modelo no conjunto de teste
y_predict = np.argmax(model.predict(x_test), axis=1)
score = accuracy_score(y_test, y_predict)

print('{}% das amostras foram classificadas corretamente!'.format(score * 100))

# Salvar o modelo
model.save('model.h5')
