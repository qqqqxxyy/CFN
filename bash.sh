#!/bin/bash
python U2net_test.py \
	--model=UNet_ssol_base2_3 \
	--model_weight=../weight/result_weight/000258_check_2_10000.pth \
	--dataset=ILSVRC --phase=2 \
	--val_property_dir=ILSVRC/data_annotation/test_list.json

python U2net_test.py \
	--model=UNet_ssol_base2_3 \
	--model_weight=../weight/result_weight/000258_check_2_15000.pth \
	--dataset=ILSVRC --phase=2 \
	--val_property_dir=ILSVRC/data_annotation/test_list.json

python U2net_test.py \
	--model=UNet_ssol_base2_3 \
	--model_weight=../weight/result_weight/000004_4800_result.pth \
	--dataset=CUB --phase=2 \
	--val_property_dir=CUB/data_annotation/test_list.json

python U2net_test.py \
	--model=UNet_ssol_base2_3 \
	--model_weight=../weight/result_weight/000258_check_2_30000.pth \
	--dataset=ILSVRC --phase=2 \
	--val_property_dir=ILSVRC/data_annotation/test_list.json
