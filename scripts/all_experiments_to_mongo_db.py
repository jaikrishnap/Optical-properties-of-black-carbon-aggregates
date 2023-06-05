import argparse
import os
import sys

import experiment_to_mongo_db


def main(args):
    logdir = os.path.realpath(args.logdir)
    for root, folders, files in os.walk(logdir):
        if os.path.isdir(os.path.join(root, '_sources')):
            args.exp_name = os.path.relpath(root, logdir)
            experiment_to_mongo_db.main(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default='../log')
    parser.add_argument('--dbhost', type=str, default='localhost')
    parser.add_argument('--dbport', type=int, default=27017)
    parser.add_argument('--dbuser', type=str, default=None)
    parser.add_argument('--dbpassword', type=str, default=None)
    args = parser.parse_args()
    main(args)
