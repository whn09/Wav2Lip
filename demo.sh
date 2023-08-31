python inference.py \
    --checkpoint_path "checkpoints/wav2lip_gan.pth" \
    --face ../VID20230620105517.mp4 \
    --audio ../test.wav \
    --outfile results/result_wav2lip-seamlessClone.mp4 \
    --resize_factor 1
