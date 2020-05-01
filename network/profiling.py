from network.ethec_experiments import ETHEC_train_model
from network.finetuner import train_cifar10
from network.oe import order_embedding_labels_with_images_train_model
import argparse


def ethec_trainer():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", help='Use DEBUG mode.', action='store_true')
    parser.add_argument("--lr", help='Input learning rate.', type=float, default=0.01)
    parser.add_argument("--batch_size", help='Batch size.', type=int, default=8)
    parser.add_argument("--evaluator", help='Evaluator type.', type=str, default='ML')
    parser.add_argument("--experiment_name", help='Experiment name.', type=str, required=True)
    parser.add_argument("--experiment_dir", help='Experiment directory.', type=str, required=True)
    parser.add_argument("--image_dir", help='Image parent directory.', type=str, required=True)
    parser.add_argument("--n_epochs", help='Number of epochs to run training for.', type=int, required=True)
    parser.add_argument("--n_workers", help='Number of workers.', type=int, default=4)
    parser.add_argument("--eval_interval", help='Evaluate model every N intervals.', type=int, default=1)
    parser.add_argument("--resume", help='Continue training from last checkpoint.', action='store_true')
    parser.add_argument("--merged", help='Use dataset which has genus and species combined.', action='store_true')
    parser.add_argument("--model", help='NN model to use. Use one of [`multi_label`, `multi_level`]',
                        type=str, required=True)
    parser.add_argument("--loss", help='Loss function to use.', type=str, required=True)
    parser.add_argument("--freeze_weights", help='This flag fine tunes only the last layer.', action='store_true')
    parser.add_argument("--set_mode", help='If use training or testing mode (loads best model).', type=str,
                        required=True)
    args = parser.parse_args(['--n_epochs', '10', '--experiment_name', 'ethec_alexnet_remove', '--experiment_dir',
                              '../exp/ethec/multi_level', '--model', 'resnet18', '--set_mode', 'train', '--debug',
                              '--image_dir', '/media/ankit/DataPartition/IMAGO_build_test_resized', '--loss',
                              'multi_level', '--eval_interval', '1', '--merged', '--lr', '0.01', # '--freeze_weights',
                              ])

    ETHEC_train_model(args)


def cifar_trainer():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", help='Use DEBUG mode.', action='store_true')
    parser.add_argument("--lr", help='Input learning rate.', type=float, default=0.01)
    parser.add_argument("--batch_size", help='Batch size.', type=int, default=8)
    parser.add_argument("--evaluator",
                        help='Evaluator type. If using `multi_level` option for --loss then is overidden.', type=str,
                        default='ML')
    parser.add_argument("--experiment_name", help='Experiment name.', type=str, required=True)
    parser.add_argument("--experiment_dir", help='Experiment directory.', type=str, required=True)
    parser.add_argument("--n_epochs", help='Number of epochs to run training for.', type=int, required=True)
    parser.add_argument("--n_workers", help='Number of workers.', type=int, default=4)
    parser.add_argument("--eval_interval", help='Evaluate model every N intervals.', type=int, default=1)
    parser.add_argument("--resume", help='Continue training from last checkpoint.', action='store_true')
    parser.add_argument("--model", help='NN model to use.', type=str, required=True)
    parser.add_argument("--loss", help='Loss function to use.', type=str, required=True)
    parser.add_argument("--freeze_weights", help='This flag fine tunes only the last layer.', action='store_true')
    parser.add_argument("--set_mode", help='If use training or testing mode (loads best model).', type=str,
                        required=True)
    cmd = """--n_epochs 10 --experiment_name cifar_hierarchical_ft_debug --experiment_dir ../exp --debug --evaluator MLST --model alexnet --loss multi_level --set_mode train"""
    args = parser.parse_args(cmd.split(' '))

    train_cifar10(args)

def ethec_emb():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", help='Use DEBUG mode.', action='store_true')
    parser.add_argument("--lr", help='Input learning rate.', type=float, default=0.01)
    parser.add_argument("--alpha", help='Margin for the loss.', type=float, default=0.05)
    parser.add_argument("--batch_size", help='Batch size.', type=int, default=8)
    parser.add_argument("--load_G_from_disk", help='If set, then loads precomputed graphs from disk.',
                        action='store_true')
    parser.add_argument("--experiment_name", help='Experiment name.', type=str, required=True)
    parser.add_argument("--experiment_dir", help='Experiment directory.', type=str, required=True)
    parser.add_argument("--load_emb_from", help='Path to embeddings .pth file', type=str, default=None)
    parser.add_argument("--load_cosine_emb", help='Path to cosine embeddings .np file', type=str, default=None)
    parser.add_argument("--image_dir", help='Image parent directory.', type=str, required=True)
    parser.add_argument("--n_epochs", help='Number of epochs to run training for.', type=int, required=True)
    parser.add_argument("--n_workers", help='Number of workers.', type=int, default=8)
    parser.add_argument("--eval_interval", help='Evaluate model every N intervals.', type=int, default=1)
    parser.add_argument("--embedding_dim", help='Dimensions of learnt embeddings.', type=int, default=10)
    parser.add_argument("--neg_to_pos_ratio", help='Number of negatives to sample for one positive.', type=int,
                        default=5)
    parser.add_argument("--resume", help='Continue training from last checkpoint.', action='store_true')
    parser.add_argument("--optimizer_method", help='[adam, sgd]', type=str, default='adam')
    parser.add_argument("--merged", help='Use dataset which has genus and species combined.', action='store_true')
    parser.add_argument("--model", help='NN model to use.', type=str, default='alexnet')
    parser.add_argument("--loss",
                        help='Loss function to use. [order_emb_loss, euc_emb_loss]',
                        type=str, required=True)
    parser.add_argument("--use_CNN", help='Use CNN.', action='store_true')
    parser.add_argument("--pick_per_level",
                        help='If set, then picks samples from each level, for the remaining, picks from images.',
                        action='store_true')
    parser.add_argument("--freeze_weights", help='This flag fine tunes only the last layer.', action='store_true')
    parser.add_argument("--half_half", help='If show 50% (label, label) and 50% (label, image) edges.',
                        action='store_true')
    parser.add_argument("--hide_levels", help='This flag shows graph levels incrementally.', action='store_true')
    parser.add_argument("--set_mode", help='If use training or testing mode (loads best model).', type=str,
                        required=True)
    parser.add_argument("--lr_step", help='List of epochs to make multiple lr by 0.1', nargs='*', default=[],
                        type=int)

    cmd = """--n_epochs 1 --use_CNN --experiment_name level_hide --experiment_dir ../exp/ethec_debug/oelwi_debug --set_mode train --image_dir /media/ankit/DataPartition/IMAGO_build_test_resized --eval_interval 1 --merged --batch_size 10 --optimizer adam --lr 0.0001 --model alexnet --loss euc_cones_loss --embedding_dim 2 --neg_to_pos_ratio 10 --alpha 0.01 --load_G_from_disk --pick_per_level --n_workers 3 --debug"""
    args = parser.parse_args(cmd.split(' '))

    order_embedding_labels_with_images_train_model(args)


if __name__ == '__main__':
    # cifar_trainer()
    # ethec_trainer()
    ethec_emb()

