from enum import IntEnum
import tensorflow as tf
from config import AUTSL_INPUT_HEIGHT, MEJIAPEREZ_INPUT_HEIGHT, WLASL100_INPUT_HEIGHT
from data_augmentation import RandomFlip, RandomScale, RandomShift, RandomRotation, RandomSpeed
from preprocessing import Center, CenterAtFirstFrame2D, FillBlueWithAngle, PadIfLessThan, RemoveZ, ResizeIfMoreThan, TranslationScaleInvariant
import tensorflow_datasets as tfds


class LayerType(IntEnum):
    Augmentation = 1
    Normalization = 2
    Data = 3


LayerDict = {
    'random_speed': {
        'type': LayerType.Augmentation,
        'layer': RandomSpeed(min_frames=40, max_frames=128, seed=5),
    },
    'random_rotation': {
        'type': LayerType.Augmentation,
        'layer': RandomRotation(factor=15.0, min_value=0.0, max_value=1.0, seed=4),
    },
    'random_flip': {
        'type': LayerType.Augmentation,
        'layer': RandomFlip("horizontal", min_value=0.0, max_value=1.0, seed=3),
    },
    'random_scale': {
        'type': LayerType.Augmentation,
        'layer': RandomScale(min_value=0.0, max_value=1.0, seed=1),
    },
    'random_shift': {
        'type': LayerType.Augmentation,
        'layer': RandomShift(min_value=0.0, max_value=1.0, seed=2)
    },
    'invariant_frame': {
        'type': LayerType.Normalization,
        'layer': TranslationScaleInvariant(level="frame")
    },
    'invariant_joint': {
        'type': LayerType.Normalization,
        'layer': TranslationScaleInvariant(level="joint")
    },
    'center': {
        'type': LayerType.Normalization,
        'layer': Center(around_index=0)
    },
    'center_at_first': {
        'type': LayerType.Normalization,
        'layer': CenterAtFirstFrame2D(around_index=0)
    },
    'train_resize': {
        'type': LayerType.Normalization,
        'layer': ResizeIfMoreThan(frames=100)
    },
    'test_resize': {
        'type': LayerType.Normalization,
        'layer': ResizeIfMoreThan(frames=100)
    },
    'pad': {
        'type': LayerType.Normalization,
        'layer': PadIfLessThan(frames=100)
    },
    'angle': {
        'type': LayerType.Data,
        'layer': FillBlueWithAngle(x_channel=0, y_channel=1, scale_to=[0, 1]),
    },
    'norm_imagenet': {
        'type': LayerType.Normalization,
        'layer': tf.keras.layers.Normalization(axis=-1,
                                               mean=[0.485, 0.456, 0.406],
                                               variance=[0.052441, 0.050176, 0.050625]),
    },
    # placeholder for layer, mean and variance are obtained dinamically
    'norm': {
        'type': LayerType.Normalization,
        'layer': tf.keras.layers.Normalization(axis=-1,
                                               mean=[248.08896, 246.56985, 0.],
                                               variance=[9022.948, 17438.518, 0.])
    }
}


# Augmentation Order = ['speed', 'rotation', 'flip', 'scale', 'shift']
PipelineDict = {
    'default': {
        'train': ['random_speed', 'random_flip', 'random_scale', 'train_resize', 'pad'],
        'test': ['test_resize', 'pad']
    },
    'default_center_at_0': {
        'train': ['random_speed', 'random_flip', 'random_scale', 'center_at_first', 'train_resize', 'pad'],
        'test': ['center_at_first', 'test_resize', 'pad']
    },
    'default_center': {
        'train': ['center', 'random_speed', 'train_resize', 'pad'],
        'test': ['center', 'test_resize', 'pad']
    },
    'default_angle': {
        'train': ['angle', 'random_speed', 'train_resize', 'pad'],
        'test': ['angle', 'test_resize', 'pad']
    },
    'invariant_frame': {
        'train': ['random_speed', 'train_resize', 'invariant_frame', 'pad'],
        'test': ['test_resize', 'invariant_frame', 'pad']
    }
}


def generate_train_dataset(dataset,
                           train_map_fn,
                           repeat=False,
                           batch_size=32,
                           buffer_size=5000,
                           deterministic=False):
    # shuffle, map and batch dataset
    if deterministic:
        train_dataset = dataset \
            .shuffle(buffer_size) \
            .map(train_map_fn) \
            .batch(batch_size)
    else:
        train_dataset = dataset \
            .shuffle(buffer_size) \
            .map(train_map_fn,
                 num_parallel_calls=tf.data.AUTOTUNE,
                 deterministic=False) \
            .batch(batch_size) \
            .prefetch(tf.data.AUTOTUNE)

    if repeat:
        train_dataset = train_dataset.repeat()

    return train_dataset


def generate_test_dataset(dataset,
                          test_map_fn,
                          batch_size=32):
    # batch dataset
    max_element_length = 200
    bucket_boundaries = list(range(1, max_element_length))
    bucket_batch_sizes = [batch_size] * max_element_length
    ds = dataset.bucket_by_sequence_length(
        element_length_func=lambda x, y: tf.shape(x)[0],
        bucket_boundaries=bucket_boundaries,
        bucket_batch_sizes=bucket_batch_sizes,
        no_padding=True)

    # map dataset
    dataset = ds \
        .map(test_map_fn,
             num_parallel_calls=tf.data.AUTOTUNE,
             deterministic=False) \
        .cache()

    return dataset


def build_pipeline(pipeline, exclude_augmentation=False, split="train"):
    # normalization: None, str or list
    if pipeline == None:
        layers = []
    elif type(pipeline) is str:
        items = [LayerDict[name] for name in PipelineDict[pipeline][split]]
        if exclude_augmentation:
            items = [item for item in items if
                     item["type"] != LayerType.Augmentation]
        layers = [item["layer"] for item in items]
    else:
        raise Exception("Pipeline " +
                        str(pipeline) + " not found")
    pipeline = tf.keras.Sequential(layers, name="normalization")
    return pipeline


class Dataset():
    def __init__(self, name: str, concat_validation_to_train=False):
        global LayerDict

        # obtain dataset
        ds, info = tfds.load(name, data_dir="datasets", with_info=True)

        # generate train dataset
        if concat_validation_to_train:
            ds["train"] = ds["train"].concatenate(ds["validation"])

        # generate norm layer
        # @tf.function
        # def extract_pose(item):
        #     return item["pose"]
        # norm = tf.keras.layers.Normalization(axis=-2)
        # norm.adapt(ds["train"].map(extract_pose,
        #                            num_parallel_calls=tf.data.AUTOTUNE))
        # LayerDict["norm"]["layer"] = norm

        # preprocess labels
        @tf.function
        def label_to_one_hot(item):
            one_hot_label = tf.one_hot(item["label"],
                                       info.features['label'].num_classes)
            return item["pose"], one_hot_label
        ds["train"] = ds["train"].map(label_to_one_hot).cache()
        ds["validation"] = ds["validation"].map(label_to_one_hot)

        # obtain characteristics of the dataset
        num_train_examples = ds["train"].cardinality()
        num_val_examples = ds["validation"].cardinality()
        if "test" in ds.keys():
            ds["test"] = ds["test"].map(label_to_one_hot)
            num_test_examples = ds["test"].cardinality()
        else:
            num_test_examples = 0
        num_total_examples = num_train_examples + num_val_examples + num_test_examples
        
        if name == "autsl_tssi":
            input_height = AUTSL_INPUT_HEIGHT
        elif name == "mejiaperez_tssi":
            input_height = MEJIAPEREZ_INPUT_HEIGHT
        elif name == "wlasl100_tssi":
            input_height = WLASL100_INPUT_HEIGHT
        else:
            raise Exception("Dataset " + name + " not found.")
            
            
        LayerDict["train_resize"]["layer"] = ResizeIfMoreThan(frames=input_height)
        LayerDict["test_resize"]["layer"] = ResizeIfMoreThan(frames=input_height)
        LayerDict["pad"]["layer"] = PadIfLessThan(frames=input_height)

        self.ds = ds
        self.name = name
        self.num_train_examples = num_train_examples
        self.num_val_examples = num_val_examples
        self.num_test_examples = num_test_examples
        self.num_total_examples = num_total_examples
        self.input_height = input_height
        self.input_width = info.features['pose'].shape[1]
        self.num_classes = info.features['label'].num_classes

    def get_training_set(self,
                         batch_size=32,
                         buffer_size=5000,
                         repeat=False,
                         deterministic=False,
                         augmentation=True,
                         pipeline="default"):
        # define pipeline
        exclude_augmentation = not augmentation
        preprocessing_pipeline = build_pipeline(
            pipeline, exclude_augmentation, "train")

        # define the train map function
        @tf.function
        def train_map_fn(x, y):
            batch = tf.expand_dims(x, axis=0)
            # batch = RemoveZ()(batch)
            batch = preprocessing_pipeline(batch, training=True)
            x = tf.ensure_shape(
                batch[0], [self.input_height, self.input_width, 3])
            return x, y

        train_ds = self.ds["train"]
        dataset = generate_train_dataset(train_ds,
                                         train_map_fn,
                                         repeat=repeat,
                                         batch_size=batch_size,
                                         buffer_size=buffer_size,
                                         deterministic=deterministic)

        return dataset

    def get_validation_set(self,
                           batch_size=32,
                           pipeline="default"):
        # define pipeline
        preprocessing_pipeline = build_pipeline(
            pipeline, exclude_augmentation=True, split="test")

        # define the val map function
        @tf.function
        def test_map_fn(batch_x, batch_y):
            # batch_x = RemoveZ()(batch_x)
            batch_x = preprocessing_pipeline(batch_x)
            return batch_x, batch_y

        val_ds = self.ds["validation"]
        dataset = generate_test_dataset(val_ds,
                                        test_map_fn,
                                        batch_size=batch_size)

        return dataset

    def get_testing_set(self,
                        batch_size=32,
                        pipeline="default"):
        if "test" not in self.ds.keys():
            return None

        # define pipeline
        preprocessing_pipeline = build_pipeline(
            pipeline, exclude_augmentation=True, split="test")

        # define the val map function
        @tf.function
        def test_map_fn(batch_x, batch_y):
            # batch_x = RemoveZ()(batch_x)
            batch_x = preprocessing_pipeline(batch_x)
            return batch_x, batch_y

        test_ds = self.ds["test"]
        dataset = generate_test_dataset(test_ds,
                                        test_map_fn,
                                        batch_size=batch_size)

        return dataset
