import argparse

def get_parsed_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', '--cws_dir', dest='cws_dir', help='directory of training csv file')
    parser.add_argument('-o', '--output', dest='output', help='directory to save training output')
    parser.add_argument('-n', '--num_cpu', dest='num_cpu', help='number of cpu for multiprocessing')
    parser.add_argument('--model_dir', dest='model_dir', help='path to model')
    parser.add_argument('--cluster', dest='cluster', help='flag to check if running on server or local',
                    action='store_true', default=False)
    parser.add_argument('-mp', dest='mp', help='flag to for multiprocessing', action='store_true', default=False)
    parser.add_argument('--run_on_batch', dest='run_on_batch', help='flag fro run_on_batch', action='store_true', default=False)
    parser.add_argument('--batch', dest='batch', help='batch', default='-1')
    args = parser.parse_args()

    return args