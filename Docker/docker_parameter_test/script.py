# script.py
import argparse

def main():
    parser = argparse.ArgumentParser(description='Process some parameters.')
    parser.add_argument('--epoch', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer type')
    parser.add_argument('--loss', type=str, default='mean_squared_error', help='Loss function')
    parser.add_argument('--metrics', nargs='+', default=['mse'], help='List of metrics')

    # For layers, we use a default value of [256, 256]
    parser.add_argument('--layers', nargs='+', type=int, default=[256, 256], help='List of layers')

    args = parser.parse_args()

    EPOCH = args.epoch
    BATCH_SIZE = args.batch_size
    LAYERS = args.layers
    OPTIMIZER = args.optimizer
    LOSS = args.loss
    METRICS = args.metrics

    print(f"EPOCH: {EPOCH}")
    print(f"BATCH_SIZE: {BATCH_SIZE}")
    print(f"LAYERS: {LAYERS}")
    print(f"OPTIMIZER: {OPTIMIZER}")
    print(f"LOSS: {LOSS}")
    print(f"METRICS: {METRICS}")

if __name__ == "__main__":
    main()