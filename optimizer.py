from tensorflow_addons.optimizers import TriangularCyclicalLearningRate
from tensorflow_addons.optimizers import SGDW
from tensorflow_addons.optimizers import AdamW
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam


def build_sgd_optimizer(initial_learning_rate=0.001,
                        maximal_learning_rate=0.01,
                        step_size=50, momentum=0.0,
                        nesterov=False, weight_decay=1e-7):
    # setup schedule
    learning_rate_schedule = TriangularCyclicalLearningRate(
        initial_learning_rate=initial_learning_rate,
        maximal_learning_rate=maximal_learning_rate,
        step_size=step_size)

    # setup the optimizer
    if weight_decay:
        initial_weight_decay = weight_decay
        maximal_weight_decay = weight_decay * \
            (maximal_learning_rate / initial_learning_rate)
        weight_decay_schedule = TriangularCyclicalLearningRate(
            initial_learning_rate=initial_weight_decay,
            maximal_learning_rate=maximal_weight_decay,
            step_size=step_size)

        optimizer = SGDW(learning_rate=learning_rate_schedule,
                         weight_decay=weight_decay_schedule,
                         momentum=momentum, nesterov=nesterov)
    else:
        optimizer = SGD(learning_rate=learning_rate_schedule,
                        momentum=momentum, nesterov=nesterov)
    return optimizer


def build_adam_optimizer(initial_learning_rate=0.001,
                         maximal_learning_rate=0.01,
                         step_size=50, epsilon=1e-7,
                         weight_decay=1e-7):
    # setup schedule
    learning_rate_schedule = TriangularCyclicalLearningRate(
        initial_learning_rate=initial_learning_rate,
        maximal_learning_rate=maximal_learning_rate,
        step_size=step_size)

    # setup the optimizer
    if weight_decay:
        initial_weight_decay = weight_decay
        maximal_weight_decay = weight_decay * \
            (maximal_learning_rate / initial_learning_rate)
        weight_decay_schedule = TriangularCyclicalLearningRate(
            initial_learning_rate=initial_weight_decay,
            maximal_learning_rate=maximal_weight_decay,
            step_size=step_size)

        optimizer = AdamW(learning_rate=learning_rate_schedule,
                          weight_decay=weight_decay_schedule,
                          epsilon=epsilon)
    else:
        optimizer = Adam(learning_rate=learning_rate_schedule,
                         epsilon=epsilon)
    return optimizer


def build_sgd_optimizer_wo_schedule(initial_learning_rate=0.001,
                                    momentum=0.0, nesterov=False):
    # setup schedule
    optimizer = SGD(learning_rate=initial_learning_rate,
                    momentum=momentum, nesterov=nesterov)
    return optimizer
