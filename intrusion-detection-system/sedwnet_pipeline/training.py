from typing import List

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


def build_callbacks(patience: int, lr_patience: int) -> List[tf.keras.callbacks.Callback]:
    return [
        EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=lr_patience, min_lr=1e-6),
    ]
