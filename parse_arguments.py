import argparse

def get_parsed_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', dest='data_dir', help='directory of the data')
    parser.add_argument('-o', '--output', dest='output', help='directory to save output')
    # parser.add_argument('-p', '--panel', dest='panel', help='panel name; it should be panel1 or panel2')
    parser.add_argument('-n', '--num_cpu', dest='num_cpu', help='number of cpu for multiprocessing')
    args = parser.parse_args()

    return args