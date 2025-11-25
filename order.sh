# Given an input folder path a, which contains an uploaded music subfolder, this code will process the data and generate a new smpl folder under path a to store the 3D SMPL parameters.
CUDA_VISIBLE_DEVICES=0 python demo_gpt.py --root_dir ./demo/1
CUDA_VISIBLE_DEVICES=1 python test_cls.py
CUDA_VISIBLE_DEVICES=2 python test_fsq.py
CUDA_VISIBLE_DEVICES=3 python test_gpt.py
