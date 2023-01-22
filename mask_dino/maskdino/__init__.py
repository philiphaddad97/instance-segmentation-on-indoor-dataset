# ------------------------------------------------------------------------
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Mask2Former https://github.com/facebookresearch/Mask2Former by Feng Li and Hao Zhang.
# ------------------------------------------------------------------------------
from . import data  # register all new datasets
from . import modeling

# config
from .config import add_maskdino_config

# dataset loading
from data.coco_instance_new_baseline_dataset_mapper import COCOInstanceNewBaselineDatasetMapper
from data.detr_dataset_mapper import DetrDatasetMapper

# models
from .maskdino import MaskDINO
# from .data.datasets_detr import coco
from .test_time_augmentation import SemanticSegmentorWithTTA

# evaluation
from .evaluation.instance_evaluation import InstanceSegEvaluator
# util
from .utils import box_ops, misc, utils
