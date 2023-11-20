# python inference.py --checkpoint_path ./checkpoints/wav2lip_gan.pth --face ./test.mp4 --audio ./test.wav --face_det_batch_size 1 --wav2lip_batch_size 1 
python inference-realtime.py --checkpoint_path ./checkpoints/wav2lip_gan.pth --face ../inputs/train_close_mouth_480p.mp4 --text "Good morning, this is a sample text to be synthesized." --face_det_batch_size 1 --wav2lip_batch_size 1 
