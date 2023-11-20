rm -rf ../inputs/train
rm -rf ../inputs/train_preprocessed

pip install -r requirements.txt

python split_video.py

python preprocess.py --data_root "../inputs" --preprocessed_root "../inputs/train_preprocessed" --batch_size 8

python generate_filelists.py

python wav2lip_train.py --data_root ../inputs/train_preprocessed --checkpoint_dir ./savedmodel --syncnet_checkpoint_path ./checkpoints/lipsync_expert.pth --checkpoint_path ./checkpoints/wav2lip.pth

python hq_wav2lip_train.py --data_root ../inputs/train_preprocessed --checkpoint_dir ./savedmodel --syncnet_checkpoint_path ./checkpoints/lipsync_expert.pth --checkpoint_path ./checkpoints/wav2lip.pth --disc_checkpoint_path ./checkpoints/visual_quality_disc.pth
