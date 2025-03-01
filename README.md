# uvr5

基于 [uvr5-cli](https://github.com/dddqmmx/uvr5-cli) 升级而来，在 windows 下很方便安装，很方便命令行调用

## 安装

```bash
python -m pip install -r requirements.txt
```

## 模型下载

下载地址:

- https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/uvr5_weights
- https://github.com/TRvlvr/model_repo/releases/tag/all_public_uvr_models

## 运行

```bash
python infer_uvr5.py -model_path uvr5_weights/HP2_all_vocals.pth -model_params 4band_v2 -audio_path ./assets/yueliang.mp3 -output_vocal_path ./separated/yueliang_vocals_2.wav -output_background_path ./separated/yueliang_background_2.wav
```

注意：
- output_vocal_path 和 output_background_path，如果不设置，则不输出对应文件
- 有的模型如果使用  4band_v2 的 model_params 报错，请换成 4band_v3
- 有的模型 output_vocal_path 和 output_background_path 是反着来的