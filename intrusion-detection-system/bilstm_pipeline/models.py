import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout, BatchNormalization, Input


def build_bilstm(input_shape: tuple[int, ...], num_classes: int) -> Model:
    model = Sequential(
        [
            Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape),
            BatchNormalization(),
            Dropout(0.3),
            Bidirectional(LSTM(64, return_sequences=True)),
            BatchNormalization(),
            Dropout(0.3),
            Bidirectional(LSTM(32)),
            BatchNormalization(),
            Dropout(0.2),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(num_classes, activation="softmax"),
        ]
    )
    return model


def build_transfer_model(base_model_path: str, num_classes: int, input_shape: tuple[int, ...]) -> Model:
    base = tf.keras.models.load_model(base_model_path, compile=False)
    if not isinstance(base, tf.keras.Model):
        raise TypeError("Loaded base model is not a Keras Model")

    if len(base.layers) < 2:
        raise ValueError("Base model has too few layers to replace classification head")

    inp = Input(shape=input_shape, name="transfer_input")
    x = inp

    for layer in base.layers[:-1]:
        x = layer(x)

    out = Dense(num_classes, activation="softmax", name="transfer_head")(x)
    return tf.keras.Model(inputs=inp, outputs=out)
