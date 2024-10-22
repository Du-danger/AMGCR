import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run AMGCR.")
    parser.add_argument('--data_path', nargs='?', default='data/', help='Input data path.')
    parser.add_argument('--seed', type=int, default=2024, help='random seed')
    parser.add_argument('--dataset', nargs='?', default='gowalla', help='Choose a dataset from {gowalla, amazon, tmall}')
    parser.add_argument('--batch_size', type=int, default=4096, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--Ks', nargs='?', default='[20, 40]', help='Metrics scale')
    parser.add_argument('--decay', default=0.99, type=float, help='learning rate')
    parser.add_argument('--note', default=None, type=str, help='note')
    parser.add_argument('--epoch', default=100, type=int, help='number of epochs')
    parser.add_argument('--d', default=32, type=int, help='embedding size')
    parser.add_argument('--l', default=2, type=int, help='number of gnn layers')
    parser.add_argument('--dropout', default=0.0, type=float, help='rate for edge dropout')
    parser.add_argument('--temp', default=0.2, type=float, help='temperature in cl loss')
    parser.add_argument('--refine', default=1, type=int, help='if refine')
    parser.add_argument('--lambda_1', default=0.2, type=float, help='weight of cl loss')
    parser.add_argument('--lambda_2', default=0.5, type=float, help='weight of pref loss')
    parser.add_argument('--lambda_3', default=1e-7, type=float, help='l2 reg weight')
    parser.add_argument('--gpu', default=0, type=int, help='the gpu to use, -1 is cpu')

    return parser.parse_args()
