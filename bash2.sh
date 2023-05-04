#!/bin/bash
python train_u2net_2.py \
	--gen_devices 0 --device=0 --seed=6 --record_file=2 \
    	--model=UNet_fb_v3 --gan_weights=../weight/pretrained_weight/BigBiGAN_x1.pth \
 	--z=../weight/pretrained_weight/embeddings/BigBiGAN_ImageNet_q_z.npy \
 	--z_noise=0.2 --bg_direction=../weight/pretrained_weight/bg_direction.pth \
 	--decay_lis 3000 3000 2500 --n_step_lis 10000 8000 10000 --batch_lis 80 50 60 \
 	--steps_per_log=1000 --steps_per_weight_save=5000 --steps_per_validation=3000 \
 	--val_root_dir=CUB/data \
 	--model_weight=../weight/pretrained_weight/000136_result.pth
 		
python train_fb_2.py \
	--gen_devices 0 --device=0 --seed=6 --record_file=2 \
    	--model=UNet_fb_v3 --gan_weights=../weight/pretrained_weight/BigBiGAN_x1.pth \
 	--z=../weight/pretrained_weight/embeddings/BigBiGAN_CUB_WSOL_train_z.npy \
 	--z_noise=0.2 --bg_direction=../weight/pretrained_weight/bg_direction.pth \
 	--decay_lis 3000 3000 2500 --n_step_lis 10000 8000 10000 --batch_lis 80 50 60 \
 	--steps_per_log=1000 --steps_per_weight_save=5000 --steps_per_validation=3000 \
 	--val_root_dir=CUB/data \
 	--model_weight=../weight/pretrained_weight/000274_40000_3_result.pth
