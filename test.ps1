./venv/Activate.ps1
# python infer_uvr5.py -model_path uvr5_weights/HP2_all_vocals.pth -model_params 4band_v2 -audio_path ./assets/yueliang.mp3 -output_vocal_path ./separated/yueliang_vocals.wav
python infer_uvr5.py -model_path uvr5_weights/HP2_all_vocals.pth -model_params 4band_v2 -audio_path ./assets/yueliang.mp3 -output_vocal_path ./separated/yueliang_vocals_2.wav -output_background_path ./separated/yueliang_background_2.wav
# python infer_uvr5.py -model_path uvr5_weights/HP3_all_vocals.pth -model_params 4band_v2 -audio_path ./assets/yueliang.mp3 -output_vocal_path ./separated/yueliang_vocals_3.wav -output_background_path ./separated/yueliang_background_3.wav
