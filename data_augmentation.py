import math
import tensorflow as tf


class RandomSpeed(tf.keras.layers.Layer):
    def __init__(self, min_frames=96, max_frames=128, seed=None, debug=False, **kwargs):
        super().__init__(**kwargs)
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.seed = seed
        self.debug = debug

    @tf.function
    def call(self, images):
        height = tf.shape(images)[1]
        width = tf.shape(images)[2]
        x_min = tf.cond(height < self.min_frames,
                        lambda: height, lambda: self.min_frames)
        x_max = self.max_frames + 1
        x = tf.random.uniform(shape=[], minval=x_min, maxval=x_max,
                              dtype=tf.int32, seed=self.seed)
        resized_images = tf.image.resize(images, [x, width])

        if self.debug:
            tf.print("speed", x)

        return resized_images


class RandomShift(tf.keras.layers.Layer):
    def __init__(self, min_value=0.0, max_value=255.0, seed=None, debug=False, **kwargs):
        super().__init__(**kwargs)
        self.min_value = min_value
        self.max_value = max_value
        self.seed = seed
        self.debug = debug

    @tf.function
    def call(self, image):
        [red, green, blue] = tf.unstack(image, axis=-1)

        left_offset = tf.abs(tf.reduce_min(red) - self.min_value)
        right_offset = tf.abs(self.max_value - tf.reduce_max(red))
        red_shift = tf.random.uniform(shape=[],
                                      minval=tf.math.negative(left_offset),
                                      maxval=right_offset,
                                      seed=self.seed)

        if self.debug:
            tf.print("red shift", red_shift)

        bottom_offset = tf.abs(tf.reduce_min(green) - self.min_value)
        top_offset = tf.abs(self.max_value - tf.reduce_max(green))
        green_shift = tf.random.uniform(shape=[],
                                        minval=tf.math.negative(bottom_offset),
                                        maxval=top_offset,
                                        seed=self.seed)

        new_red = tf.add(red, red_shift)
        new_green = tf.add(green, green_shift)

        if self.debug:
            tf.print("green shift", green_shift)

        return tf.stack([new_red, new_green, blue], axis=-1)


class RandomFlip(tf.keras.layers.Layer):
    def __init__(self, mode, min_value=0.0, max_value=255.0, around_zero=False, seed=None, debug=False, **kwargs):
        super().__init__(**kwargs)
        self.mode = mode
        self.min_value = min_value
        self.max_value = max_value
        self.seed = seed
        self.debug = debug
        self.around_zero = tf.constant(around_zero)

    @tf.function
    def add_factor(self, channel):
        # channel.shape => (examples, frames, joints)
        # channel_maxs.shape => (examples, 1, 1)
        # channel max per example
        channel_max = tf.reduce_max(channel, axis=[-1, -2], keepdims=True)
        channel_min = tf.reduce_min(channel, axis=[-1, -2], keepdims=True)
        channel_mid = (channel_max + channel_min) / 2
        return channel_mid * 2

    @tf.function
    def call(self, image):
        rand = tf.random.uniform(shape=[],
                                 minval=0.,
                                 maxval=1.,
                                 seed=self.seed)
        [red, green, blue] = tf.unstack(image, axis=-1)
        flip_horizontal = tf.logical_and(
            rand > 0.5, tf.equal(self.mode, 'horizontal'))
        flip_vertical = tf.logical_and(
            rand > 0.5, tf.equal(self.mode, 'vertical'))
        zeros = tf.zeros(tf.shape(red))
        flip_horizontal_around_mid = tf.logical_and(
            flip_horizontal, tf.math.logical_not(self.around_zero))
        flip_vertical_around_mid = tf.logical_and(
            flip_vertical, tf.math.logical_not(self.around_zero))
        red_add_factor = tf.cond(
            flip_horizontal_around_mid, lambda: self.add_factor(red), lambda: zeros)
        green_add_factor = tf.cond(
            flip_vertical_around_mid, lambda: self.add_factor(green), lambda: zeros)
        new_red = tf.cond(
            flip_horizontal, lambda: tf.add(-red, red_add_factor), lambda: red)
        new_green = tf.cond(
            flip_vertical, lambda: tf.add(-green, green_add_factor), lambda: green)

        if self.debug:
            tf.print("flip", rand)

        return tf.stack([new_red, new_green, blue], axis=-1)


class RandomRotation(tf.keras.layers.Layer):
    def __init__(self, factor=15.0, min_value=0.0, max_value=255.0, around_zero=False, clip=True, seed=None, debug=False, **kwargs):
        super().__init__(**kwargs)
        self.min_degree = tf.math.negative(factor)
        self.max_degree = factor
        self.min_value = min_value
        self.max_value = max_value
        self.seed = seed
        self.debug = debug
        self.around_zero = tf.constant(around_zero)
        self.clip = tf.constant(clip)

    @tf.function
    def red_origin(self, red):
        red_maxs = tf.reduce_max(red, axis=-1, keepdims=True)
        red_mins = tf.reduce_min(red, axis=-1, keepdims=True)
        return (red_maxs + red_mins) / 2

    @tf.function
    def green_origin(self, green):
        # option #1 middle of green
        # green_maxs = tf.reduce_max(green, axis=-1, keepdims=True)
        # green_mins = tf.reduce_min(green, axis=-1, keepdims=True)
        # return (green_maxs + green_mins) / 2

        # option #2 max of green because lower body is closer to 1 than to 0
        green_maxs = tf.reduce_max(green, axis=-1, keepdims=True)
        return green_maxs

    @tf.function
    def call(self, image):
        degree = tf.random.uniform(shape=[],
                                   minval=self.min_degree,
                                   maxval=self.max_degree,
                                   seed=self.seed)
        if self.debug:
            tf.print("degree", degree)

        angle = degree * math.pi / 180.0
        [red, green, blue] = tf.unstack(image, axis=-1)

        red_origin = tf.cond(self.around_zero,
                             lambda: tf.zeros(tf.shape(red)),
                             lambda: self.red_origin(red))
        green_origin = tf.cond(self.around_zero,
                               lambda: tf.zeros(tf.shape(green)),
                               lambda: self.green_origin(green))

        new_red = red_origin + \
            tf.math.cos(angle) * (red - red_origin) - \
            tf.math.sin(angle) * (green - green_origin)
        new_green = green_origin + \
            tf.math.sin(angle) * (red - red_origin) + \
            tf.math.cos(angle) * (green - green_origin)

        new_red = tf.cond(
            self.clip,
            lambda: tf.clip_by_value(new_red, self.min_value, self.max_value),
            lambda: new_red)
        new_green = tf.cond(
            self.clip,
            lambda: tf.clip_by_value(
                new_green, self.min_value, self.max_value),
            lambda: new_green)

        return tf.stack([new_red, new_green, blue], axis=-1)


class RandomVerticalStretch(tf.keras.layers.Layer):
    def __init__(self, min_value=0.0, max_value=255.0, seed=None, debug=False, **kwargs):
        super().__init__(**kwargs)
        self.min_value = min_value
        self.max_value = max_value
        self.seed = seed
        self.debug = debug

    @tf.function
    def round_down_float_to_1_decimal(self, num):
        return tf.math.floor(num * 10.0) / 10.0

    @tf.function
    def call(self, image):
        [red, green, blue] = tf.unstack(image, axis=-1)

        green_maxs = tf.reduce_max(green, axis=-1, keepdims=True)
        green_mins = tf.reduce_min(green, axis=-1, keepdims=True)
        green_mids = (green_maxs + green_mins) / 2
        green_alphas_1 = tf.abs(
            (green_mids - self.min_value) / (green_mids - green_mins))
        green_alphas_2 = tf.abs(
            (self.max_value - green_mids) / (green_maxs - green_mids))
        green_alpha = self.round_down_float_to_1_decimal(
            tf.reduce_min([green_alphas_1, green_alphas_2]))

        max_alpha = tf.maximum(tf.reduce_min([green_alpha]), 0.5)
        alpha = tf.random.uniform(
            shape=[], minval=0.5, maxval=max_alpha, seed=self.seed)
        new_green = alpha * (green - green_mids) + green_mids

        if self.debug:
            tf.print("alpha", alpha, "max_alpha", max_alpha)

        return tf.stack([red, new_green, blue], axis=-1)


class RandomHorizontalStretch(tf.keras.layers.Layer):
    def __init__(self, min_value=0.0, max_value=255.0, seed=None, debug=False, **kwargs):
        super().__init__(**kwargs)
        self.min_value = min_value
        self.max_value = max_value
        self.seed = seed
        self.debug = debug

    @tf.function
    def round_down_float_to_1_decimal(self, num):
        return tf.math.floor(num * 10.0) / 10.0

    @tf.function
    def call(self, image):
        [red, green, blue] = tf.unstack(image, axis=-1)

        red_maxs = tf.reduce_max(red, axis=-1, keepdims=True)
        red_mins = tf.reduce_min(red, axis=-1, keepdims=True)
        red_mids = (red_maxs + red_mins) / 2
        red_alphas_1 = tf.abs(
            (red_mids - self.min_value) / (red_mids - red_mins))
        red_alphas_2 = tf.abs(
            (self.max_value - red_mids) / (red_maxs - red_mids))
        red_alpha = self.round_down_float_to_1_decimal(
            tf.reduce_min([red_alphas_1, red_alphas_2]))

        max_alpha = tf.maximum(tf.reduce_min([red_alpha]), 0.5)
        alpha = tf.random.uniform(
            shape=[], minval=0.5, maxval=max_alpha, seed=self.seed)
        new_red = alpha * (red - red_mids) + red_mids

        if self.debug:
            tf.print("alpha", alpha, "max_alpha", max_alpha)

        return tf.stack([new_red, green, blue], axis=-1)


class RandomScale(tf.keras.layers.Layer):
    def __init__(self, min_value=0.0, max_value=255.0, seed=None, debug=False, **kwargs):
        super().__init__(**kwargs)
        self.min_value = min_value
        self.max_value = max_value
        self.seed = seed
        self.debug = debug

    @tf.function
    def round_down_float_to_1_decimal(self, num):
        return tf.math.floor(num * 10.0) / 10.0

    @tf.function
    def call(self, batch):
        # batch.shape => [examples, frames, joints, coordinates]
        # [red, green, blue].shape => [examples, frames, joints]
        [red, green, blue] = tf.unstack(batch, axis=-1)

        # [color]_max/min/mid.shape => [examples]
        red_max = tf.reduce_max(tf.reduce_max(red, axis=-1), axis=-1)
        red_min = tf.reduce_min(tf.reduce_min(red, axis=-1), axis=-1)
        red_mid = (red_max + red_min) / 2
        green_max = tf.reduce_max(tf.reduce_max(green, axis=-1), axis=-1)
        green_min = tf.reduce_min(tf.reduce_min(green, axis=-1), axis=-1)
        green_mid = (green_max + green_min) / 2

        # [color]_centered.shape => [examples, frames, joints]
        red_centered = red - red_mid
        green_centered = green - green_mid

        # [color]_max_allowed_value.shape => [examples]
        red_max_allowed_value = tf.minimum(
            tf.abs(self.max_value - red_mid),
            tf.abs(red_mid - self.min_value))
        green_max_allowed_value = tf.minimum(
            tf.abs(self.max_value - green_mid),
            tf.abs(green_mid - self.min_value))

        # [color]_scale.shape => [examples]
        red_centered_max = tf.reduce_max(tf.reduce_max(
            tf.abs(red_centered), axis=-1), axis=-1)
        red_max_scale = self.round_down_float_to_1_decimal(
            red_max_allowed_value / red_centered_max)
        green_centered_max = tf.reduce_max(tf.reduce_max(
            tf.abs(green_centered), axis=-1), axis=-1)
        green_max_scale = self.round_down_float_to_1_decimal(
            green_max_allowed_value / green_centered_max)

        # max_alpha.shape => [examples]
        max_alpha = tf.maximum(tf.reduce_min(
            [red_max_scale, green_max_scale], axis=0), 0.5)

        # alpha.shape => [examples]
        alpha = tf.random.uniform(shape=tf.shape(
            max_alpha), minval=0.5, maxval=max_alpha, seed=self.seed)

        if self.debug:
            tf.print("alpha", alpha)

        # new_[color].shape: (examples, frames, joints)
        new_red = alpha * (red - red_mid) + red_mid
        new_green = alpha * (green - green_mid) + green_mid

        return tf.stack([new_red, new_green, blue], axis=-1)
