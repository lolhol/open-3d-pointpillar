import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import open3d as o3d
import numpy as np
from open3d.ml.tf import datasets
from open3d.ml.tf import pipelines
import open3d.ml.tf as ml3d
import open3d.ml as _ml3d
import open3d as o3d
import tensorflow as tf
import keras.optimizers.legacy as optimizers
from util import TrainingPipeline
from util import ArgsParser
from util import visualize

tf.random.set_seed(42)

def main():
    parser = ArgsParser.CustomArgsParser()
    parsed_args = parser.parse()

    if parsed_args["visualize"]:
        visualize.open3d_point_cloud(visualize.to_open3d_point_cloud(visualize.load_kitti_from_file(parsed_args["bin_path"])), window_name=parsed_args["vis_name"])
    elif parsed_args["train"] or parsed_args["test"]:
        cfg = _ml3d.utils.Config.load_from_file(parsed_args["cfg_file"])

        optimizer = optimizers.Adam(learning_rate=0.001)
        model = ml3d.models.PointPillars(optimizer=optimizer, **cfg.model)

        pipeline = pipelines.ObjectDetection(model=model, **cfg.pipeline)

        if parsed_args["test"]:
            pipeline.load_ckpt(parsed_args["ckpt_path"])

            point_cloud_data = visualize.load_kitti_from_file(parsed_args["bin_path"])                
            o3d_cloud = visualize.to_open3d_point_cloud(point_cloud_data)
            results = pipeline.run_inference(o3d_cloud)
            visualize.open3d_point_cloud(point_cloud_data, results=results, window_name="Inference Results")

        elif parsed_args["train"]:
            pipeline.dataset = ml3d.datasets.KITTI(**cfg.dataset)
            pipeline.run_train()
            pipeline.save_ckpt() # to be sure
 
if __name__ == "__main__":
    main()