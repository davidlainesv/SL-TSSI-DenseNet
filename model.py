import tensorflow as tf
from efficient_net_b0 import EfficientNetB0
from loss import select_loss
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications.densenet import DenseNet121
# from tensorflow.keras.applications.efficientnet import EfficientNetB0


def build_densenet121_model(input_shape=[None, 135, 2],
                            dropout=0,
                            optimizer=None,
                            pretraining=False,
                            use_loss="crossentroypy",
                            num_classes=None):
    # setup backbone
    weights = 'imagenet' if pretraining else None
    backbone = DenseNet121(input_shape=input_shape,
                           weights=weights,
                           include_top=False,
                           pooling="avg")

    # setup model
    inputs = Input(shape=input_shape)
    x = backbone(inputs)
    x = Dropout(dropout)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=predictions)

    # setup the metrics
    metrics = [
        TopKCategoricalAccuracy(k=1, name='top_1', dtype=tf.float32)
    ]

    # setup the loss
    loss = select_loss(use_loss)

    # compile the model
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


def build_efficientnet_model(input_shape=[None, 128, 3],
                             dropout=0,
                             optimizer=None,
                             pretraining=True,
                             use_loss="crossentropy",
                             num_classes=100):
    # setup backbone
    weights = "imagenet" if pretraining else None
    backbone = EfficientNetB0(input_shape=input_shape,
                              weights=weights,
                              include_top=False,
                              pooling="avg")

    # setup model
    inputs = Input(shape=input_shape)
    x = backbone(inputs)
    x = Dropout(dropout)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=predictions)

    # setup the metrics
    metrics = [
        TopKCategoricalAccuracy(k=1, name='top_1', dtype=tf.float32)
    ]

    # setup the model
    loss = select_loss(use_loss)

    # compile the model
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model
