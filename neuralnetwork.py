import numpy as np
import tensorflow as tf
from sklearn.datasets import make_moons

X = np.array(
    [
        [-2, 6],
        [1, 7],
        [3, 6],
        [6, 2],
        [6, -1],
        [4, -2],
        [2, -4],
        [0, -5],
        [-2, -3],
        [-4, -1],
        [-8, -2],
        [-7, -4],
    ]
)
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
y = tf.keras.utils.to_categorical(y, num_classes=4)

print(X.shape)  # Esperado: (12, 2)
print(y.shape)  # Esperado: (12, 4)

model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(4, activation="tanh", input_shape=(2,)),
        tf.keras.layers.Dense(4, activation="softmax"),
    ]
)

model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.5),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"],
)

model.fit(X, y, epochs=15, batch_size=1)

p = model.predict(
    np.array(
        [
            [-2, 3],
            [1, 5],
            [1, 7],
            [10, 5],
            [2, 1],
            [8, -4],
            [-2, -4],
            [-3, -6],
            [-1, -8],
            [-8, -2],
            [-8, 2],
            [-5, -1],
        ]
    )
)

print("hola")
print(p)
