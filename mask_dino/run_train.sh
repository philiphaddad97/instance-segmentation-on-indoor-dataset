python train_net.py \
--output_dir output_nipg1_lr00001_epochs+10_01 \
--dataset_path /home/ADND_J1/instance-segmentation-on-indoor-dataset/coco/ \
--num-gpus 1 \
--config-file /home/ADND_J1/instance-segmentation-on-indoor-dataset/mask_dino/configs/maskdino_R50_bs16_50ep_3s.yaml \
MODEL.WEIGHTS /home/ADND_J1/instance-segmentation-on-indoor-dataset/mask_dino/pretrained_models/model_lr00001_epochs10.pth