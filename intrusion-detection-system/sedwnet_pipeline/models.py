import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    BatchNormalization,
    Activation,
    Add,
    Multiply,
    Reshape,
    Flatten,
    LayerNormalization,
    Conv1D,
    SeparableConv1D,
    GlobalAveragePooling1D,
)


def build_attention_resnet_mlp(
    input_dim: int,
    num_classes: int,
    *,
    use_feature_mha: bool,
    mha_heads: int,
    mha_key_dim: int,
    mha_dropout: float,
) -> Model:
    inputs = Input(shape=(input_dim,), name="tabular_input")

    if use_feature_mha:
        tokens = Reshape((input_dim, 1), name="feat_tokens")(inputs)
        attn_out = tf.keras.layers.MultiHeadAttention(
            num_heads=mha_heads,
            key_dim=mha_key_dim,
            dropout=mha_dropout,
            name="feat_mha",
        )(tokens, tokens)
        tokens = Add(name="feat_mha_residual")([tokens, attn_out])
        tokens = LayerNormalization(name="feat_mha_norm")(tokens)
        x0 = Flatten(name="feat_mha_flatten")(tokens)
    else:
        x0 = inputs

    x = Dense(256, kernel_initializer="he_normal")(x0)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.3)(x)

    res1 = x
    x = Dense(256, kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.3)(x)

    x = Dense(256, kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)

    gate = Dense(256, activation="relu", kernel_initializer="he_normal")(x)
    gate = Dense(256, activation="sigmoid", kernel_initializer="he_normal")(gate)
    x = Multiply()([x, gate])

    x = Add()([x, res1])
    x = Activation("relu")(x)

    outputs = Dense(num_classes, activation="softmax", dtype="float32")(x)
    return Model(inputs, outputs, name="ResNet_Attn_FeatMHA_CyberSec")


def _se_block_1d(x, reduction: int = 16):
    channels = int(x.shape[-1])
    squeeze = GlobalAveragePooling1D()(x)
    squeeze = Reshape((1, channels))(squeeze)
    hidden = max(channels // reduction, 4)
    excite = Dense(hidden, activation="relu", kernel_initializer="he_normal", use_bias=False)(squeeze)
    excite = Dense(channels, activation="sigmoid", kernel_initializer="he_normal", use_bias=False)(excite)
    return Multiply()([x, excite])


def _sedwnet_block(x, filters: int, stride: int = 1, se_reduction: int = 16, dropout: float = 0.0):
    residual = x

    if stride != 1 or int(x.shape[-1]) != filters:
        residual = Conv1D(filters, 1, strides=stride, padding="same", kernel_initializer="he_normal")(residual)
        residual = BatchNormalization()(residual)

    x = SeparableConv1D(
        filters,
        3,
        strides=stride,
        padding="same",
        depthwise_initializer="he_normal",
        pointwise_initializer="he_normal",
    )(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = SeparableConv1D(
        filters,
        3,
        strides=1,
        padding="same",
        depthwise_initializer="he_normal",
        pointwise_initializer="he_normal",
    )(x)
    x = BatchNormalization()(x)

    x = _se_block_1d(x, reduction=se_reduction)
    if dropout and dropout > 0:
        x = Dropout(dropout)(x)

    x = Add()([x, residual])
    x = Activation("relu")(x)
    return x


def build_se_dwnet(input_dim: int, num_classes: int) -> Model:
    inputs = Input(shape=(input_dim,), name="tabular_input")
    x = Reshape((input_dim, 1), name="feat_as_1d")(inputs)

    x = Conv1D(64, 3, padding="same", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = _sedwnet_block(x, filters=64, stride=1)
    x = _sedwnet_block(x, filters=128, stride=2)
    x = _sedwnet_block(x, filters=256, stride=2)

    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation="softmax", dtype="float32")(x)
    return Model(inputs, outputs, name="SE_DWNet_CyberSec")


def build_transfer_model(base_model_path: str, num_classes: int) -> Model:
    base = tf.keras.models.load_model(base_model_path, compile=False)

    if not isinstance(base, tf.keras.Model):
        raise TypeError("Loaded base model is not a Keras Model")

    if len(base.layers) < 2:
        raise ValueError("Base model has too few layers to replace classification head")

    feature_extractor = tf.keras.Model(inputs=base.input, outputs=base.layers[-2].output)
    x = feature_extractor.output
    out = Dense(num_classes, activation="softmax", name="transfer_head")(x)
    return tf.keras.Model(inputs=feature_extractor.input, outputs=out)
