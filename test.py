import argparse
from config import RANDOM_SEED
from dataset import Dataset
import numpy as np
import wandb
from wandb.keras import WandbCallback
import tensorflow as tf
from model import build_densenet121_model
from optimizer import build_sgd_optimizer
from utils import str2bool

dataset = None


def run_experiment(config=None, log_to_wandb=True, verbose=0):
    global dataset

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    # check if config was provided
    if config is None:
        raise Exception("Not config provided.")
    print("[INFO] Configuration:", config, "\n")

    # check if dataset was provided
    if dataset is None:
        raise Exception("Dataset not provided.")

    # generate testing dataset
    testing_dataset = dataset.get_testing_set(
        batch_size=config['batch_size'],
        pipeline=config['pipeline'])

    # describe dataset distribution
    print("[INFO] Dataset Total examples:", dataset.num_total_examples)
    print("[INFO] Dataset Testing examples:", dataset.num_test_examples)

    # describe input shape
    input_shape = [dataset.input_height, dataset.input_width, 3]
    print("[INFO] Input Shape:", input_shape)

    # setup optimizer
    optimizer = build_sgd_optimizer(initial_learning_rate=config['initial_learning_rate'],
                                    maximal_learning_rate=config['maximal_learning_rate'],
                                    momentum=config['momentum'],
                                    nesterov=config['nesterov'],
                                    step_size=config['step_size'],
                                    weight_decay=config['weight_decay'])
    
    # setup model
    model = build_densenet121_model(input_shape=input_shape,
                                    dropout=config['dropout'],
                                    optimizer=optimizer,
                                    pretraining=config['pretraining'])
    
    # load weights
    model.load_weights(config['weights_dir'])

    # print summary of the model
    model.summary()

    # setup callbacks
    callbacks = []
    if log_to_wandb:
        wandb_callback = WandbCallback(
            monitor="val_top_1",
            mode="max",
            save_model=False
        )
        callbacks.append(wandb_callback)

    # train model
    scores = model.evaluate(testing_dataset, return_dict=True)

    # get the logs of the evaluation
    return scores


def agent_fn(config, project, entity, verbose=0):
    wandb.init(entity=entity, project=project, config=config,
               reinit=True, settings=wandb.Settings(code_dir="."))
    _ = run_experiment(config=wandb.config, log_to_wandb=True, verbose=verbose)
    wandb.finish()


def main(args):
    global dataset

    dataset = Dataset(args.dataset)
    steps_per_epoch = np.ceil(dataset.num_train_examples / args.batch_size)
    config = {
        'pretraining': args.pretraining,
        'dropout': args.dropout,

        'initial_learning_rate': args.lr_min,
        'maximal_learning_rate': args.lr_max,
        'momentum': 0.9,
        'nesterov': True,
        'weight_decay': args.weight_decay,
        'step_size': int(args.num_epochs / 2) * steps_per_epoch,

        'augmentation': args.augmentation,
        'batch_size': args.batch_size,
        'pipeline': args.pipeline,
        'num_epochs': args.num_epochs,
        
        'weights_dir': args.weights_dir
    }

    project_name = args.dataset + "_" + args.project
    agent_fn(config=config, entity=args.entity, project=project_name, verbose=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--entity', type=str,
                        help='Entity', default='davidlainesv')
    parser.add_argument('--project', type=str,
                        help='Project name', default='testing')
    parser.add_argument('--dataset', type=str,
                        help='Name of dataset', default='wlasl100')
    parser.add_argument('--weights_dir', type=str,
                        help='Dir to weights', required=True)

    parser.add_argument('--dropout', type=float,
                        help='Dropout at the final layer', default=0)
    
    parser.add_argument('--lr_min', type=float,
                        help='Minimum learning rate', default=0.001)
    parser.add_argument('--lr_max', type=float,
                        help='Minimum learning rate', default=0.01)
    parser.add_argument('--weight_decay', type=float,
                        help='Weight decay', default=0)
    
    parser.add_argument('--batch_size', type=int,
                        help='Batch size of training and testing', default=64)
    parser.add_argument('--pipeline', type=str,
                        help='Pipeline', default="default")
    parser.add_argument('--num_epochs', type=int,
                        help='Number of epochs', default=24)

    args = parser.parse_args()

    print(args)

    main(args)
