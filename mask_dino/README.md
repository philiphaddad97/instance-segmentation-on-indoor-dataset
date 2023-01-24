# Installation
```
pip install -U opencv-python

# under your working directory
git clone git@github.com:facebookresearch/detectron2.git
cd detectron2
pip install -e .
pip install git+https://github.com/cocodataset/panopticapi.git
pip install git+https://github.com/mcordts/cityscapesScripts.git

cd mask_dino
pip install -r requirements.txt
cd maskdino/modeling/pixel_decoder/ops
sh make.sh
```

# Running

Under ```mask_dino``` directory there are two ```.sh``` files:
one for run the train ```run_train.sh``` and
the other to run the run the evaluation ```run_eval.sh```

Note: To modify the model, training configs they can be find in ```mask_dino/configs```