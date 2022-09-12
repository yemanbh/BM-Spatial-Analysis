import argparse

def get_parsed_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--csv_dir', dest='csv_dir', help='directory of csv files')
    parser.add_argument('-d', '--cws_dir', dest='cws_dir', help='directory of cws images')
    parser.add_argument('-o', '--output_dir', dest='output_dir', help='saving dir')
    parser.add_argument('-s', '--src_dir', dest='src_dir', help='path to the code')
    parser.add_argument('--num_cpu', dest='num_cpu', help='number of cpu for multiprocessing')
    parser.add_argument('--cluster', dest='cluster', help='flag to check if running on server or local',
                        action='store_true', default=False)
    parser.add_argument('--mp', dest='mp', help='flag for multi-processing',
                        action='store_true', default=False)
    parser.add_argument('--batch', dest='batch', help='1-N, N=number of batches/folders. -1 (all folders)')
    args = parser.parse_args()

    return args
