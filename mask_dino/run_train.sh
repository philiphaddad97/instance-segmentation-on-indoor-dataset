python train_net.py \
--output_dir output_lr00001_maxiter401916_epochs12
--dataset_path /home/ADND_J1/instance-segmentation-on-indoor-dataset/coco
--num-gpus 1 \
--config-file /home/ADND_J1/instance-segmentation-on-indoor-dataset/mask_dino/configs/maskdino_R50_bs16_50ep_3s.yaml \
MODEL.WEIGHTS /home/ADND_J1/MaskDINO/pre_trained_models/maskdino_r50_50ep_300q_hid1024_3sd1_instance_maskenhanced_mask46.1ap_box51.5ap.pth