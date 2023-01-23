import os
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo
from detectron2.config import get_cfg

project_path = os.getcwd() + '/instance-segmentation-on-indoor-dataset'
dataset_path = project_path + '/coco/'
annotations_path = dataset_path + 'annotations/'
register_coco_instances("coco_train", {}, annotations_path + "instances_train2017.json", dataset_path + "train2017")
register_coco_instances("coco_valid", {}, annotations_path + "instances_val2017.json", dataset_path + "val2017")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("coco_train",)
cfg.DATASETS.TEST = ("coco_valid",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml") 
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 10000 
cfg.SOLVER.STEPS = []
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 9 
cfg.TEST.EVAL_PERIOD = 0 