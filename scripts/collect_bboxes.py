import logging
import argparse
import pickle
import numpy as np
import multiprocessing

from tqdm import tqdm
from os.path import join
from open3d.ml.datasets import utils
from open3d.ml import datasets
from functools import partial


def parse_args():
    parser = argparse.ArgumentParser(
        description='Collect bounding boxes for augmentation.')
    parser.add_argument('--dataset_path',
                        help='Path to dataset root',
                        required=True)
    parser.add_argument(
        '--out_path',
        help='Output path to store pickle (defaults to dataset_path)',
        default=None,
        required=False)
    parser.add_argument('--dataset_type',
                        help='Name of dataset class',
                        default="KITTI",
                        required=False)
    parser.add_argument('--num_cpus',
                        help='Number of threads to use.',
                        type=int,
                        default=multiprocessing.cpu_count(),
                        required=False)
    parser.add_argument(
        '--max_pc',
        help='Limit on the number of point clouds to process. Default is None (process the entire dataset).',
        type=int,
        default=None,
        required=False)

    args = parser.parse_args()

    dict_args = vars(args)
    for k in dict_args:
        v = dict_args[k]
        print(f"{k}: {v}" if v is not None else f"{k} not given")

    return args


def process_boxes(i, train):
    data = train.get_data(i)
    bbox = data['bounding_boxes']
    flat_bbox = [box.to_xyzwhlr() for box in bbox]
    indices = utils.operations.points_in_box(data['point'], flat_bbox)
    bboxes = []
    for i, box in enumerate(bbox):
        pts = data['point'][indices[:, i]]
        box.points_inside_box = pts
        bboxes.append(box)
    return bboxes


if __name__ == '__main__':
    """Collect bounding boxes for augmentation.

    This script constructs a bbox dictionary for later data augmentation.

    Args:
        dataset_path (str): Directory to load dataset data.
        out_path (str): Directory to save pickle file (infos).
        dataset_type (str): Name of dataset object under `ml3d/datasets` to use 
                            to load the test data split from the dataset folder.
                            Uses reflection to dynamically import this dataset 
                            object by name. Default: KITTI.
        num_cpus (int): Number of threads to use. Default is all CPU cores.

    Example usage:

    python scripts/collect_bboxes.py --dataset_path /path/to/data --dataset_type KITTI
    """

    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(asctime)s - %(module)s - %(message)s',
    )

    args = parse_args()
    out_path = args.out_path if args.out_path else args.dataset_path
    dataset_class = getattr(datasets, args.dataset_type)
    dataset = dataset_class(args.dataset_path)
    train = dataset.get_split('train')

    max_pc = len(train) if args.max_pc is None else args.max_pc

    rng = np.random.default_rng()
    query_pc = range(len(train)) if max_pc >= len(train) else rng.choice(
        range(len(train)), max_pc, replace=False)

    print(f"Found {len(train)} training samples, using {max_pc}.")
    print(f"Using {args.num_cpus} CPUs. This may take a few minutes...")

    # Use multiprocessing with the train dataset passed into process_boxes
    with multiprocessing.Pool(args.num_cpus) as p:
        process_fn = partial(process_boxes, train=train)
        bboxes = list(tqdm(p.imap(process_fn, query_pc), total=len(query_pc)))

    # Flatten the list of bounding boxes
    bboxes = [e for l in bboxes for e in l]

    # Save the bounding boxes to a pickle file
    with open(join(out_path, 'bboxes.pkl'), 'wb') as file:
        pickle.dump(bboxes, file)
        print(f"Saved {len(bboxes)} bounding boxes to {out_path}/bboxes.pkl.")