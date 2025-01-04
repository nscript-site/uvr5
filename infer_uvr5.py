import os, sys, torch, warnings, pdb
now_dir = os.getcwd()
sys.path.append(now_dir)
from json import load as ll
warnings.filterwarnings("ignore")
import librosa
import importlib
import numpy as np
import hashlib, math
from tqdm import tqdm
from uvr5_pack.lib_v5 import spec_utils
from uvr5_pack.utils import _get_name_params, inference
from uvr5_pack.lib_v5.model_param_init import ModelParameters
import soundfile as sf
from uvr5_pack.lib_v5.nets_new import CascadedNet
from uvr5_pack.lib_v5 import nets_61968KB as nets
import argparse

def ensure_directory_exists(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

class AudioSeparator:
    def __init__(self, agg, model_path, device, is_half, model_params):
        self.model_path = model_path
        self.device = device
        self.data = {
            # Processing Options
            "postprocess": False,
            "tta": False,
            # Constants
            "window_size": 512,
            "agg": agg,
            "high_end_process": "mirroring",
        }
        if model_params == "4band_v3":
            mp = ModelParameters("uvr5_pack/lib_v5/modelparams/4band_v3.json")
            nout = 64 if "DeReverb" in model_path else 48
            model = CascadedNet(mp.param["bins"] * 2, nout)
        if model_params == "4band_v2":
            mp = ModelParameters("uvr5_pack/lib_v5/modelparams/4band_v2.json")
            model = nets.CascadedASPPNet(mp.param["bins"] * 2)
        cpk = torch.load(model_path, map_location="cpu")
        model.load_state_dict(cpk)
        model.eval()
        if is_half:
            model = model.half().to(device)
        else:
            model = model.to(device)

        self.mp = mp
        self.model = model
    def separate(self, music_file, vocal_file_path=None, ins_file_path=None, model_params=None, format="flac"):#3个VR模型vocal和ins是反的
        name = os.path.basename(music_file)

        if ins_file_path is not None and ins_file_path != '':
            ensure_directory_exists(ins_file_path)

        if vocal_file_path is not None and vocal_file_path != '':
            ensure_directory_exists(vocal_file_path)

        X_wave, y_wave, X_spec_s, y_spec_s = {}, {}, {}, {}
        bands_n = len(self.mp.param["band"])
        # print(bands_n)
        for d in range(bands_n, 0, -1):
            bp = self.mp.param["band"][d]
            if d == bands_n:  # high-end band
                (
                    X_wave[d],
                    _,
                ) = librosa.core.load(  # 理论上librosa读取可能对某些音频有bug，应该上ffmpeg读取，但是太麻烦了弃坑
                    music_file,
                    sr = bp["sr"],
                    mono = False,
                    dtype=np.float32,
                    res_type=bp["res_type"],
                )
                if X_wave[d].ndim == 1:
                    X_wave[d] = np.asfortranarray([X_wave[d], X_wave[d]])
            else:  # lower bands
                X_wave[d] = librosa.core.resample(
                    X_wave[d + 1],
                    orig_sr = self.mp.param["band"][d + 1]["sr"],
                    target_sr = bp["sr"],
                    res_type=bp["res_type"],
                )
            # Stft of wave source
            X_spec_s[d] = spec_utils.wave_to_spectrogram_mt(
                X_wave[d],
                bp["hl"],
                bp["n_fft"],
                self.mp.param["mid_side"],
                self.mp.param["mid_side_b2"],
                self.mp.param["reverse"],
            )
            # pdb.set_trace()
            if d == bands_n and self.data["high_end_process"] != "none":
                input_high_end_h = (bp["n_fft"] // 2 - bp["crop_stop"]) + (
                    self.mp.param["pre_filter_stop"] - self.mp.param["pre_filter_start"]
                )
                input_high_end = X_spec_s[d][
                    :, bp["n_fft"] // 2 - input_high_end_h : bp["n_fft"] // 2, :
                ]

        X_spec_m = spec_utils.combine_spectrograms(X_spec_s, self.mp)
        aggresive_set = float(self.data["agg"] / 100)
        aggressiveness = {
            "value": aggresive_set,
            "split_bin": self.mp.param["band"][1]["crop_stop"],
        }
        with torch.no_grad():
            pred, X_mag, X_phase = inference(
                X_spec_m, self.device, self.model, aggressiveness, self.data
            )
        # Postprocess
        if self.data["postprocess"]:
            pred_inv = np.clip(X_mag - pred, 0, np.inf)
            pred = spec_utils.mask_silence(pred, pred_inv)
        y_spec_m = pred * X_phase
        v_spec_m = X_spec_m - y_spec_m

        if ins_file_path is not None and ins_file_path != '':
            if self.data["high_end_process"].startswith("mirroring"):
                input_high_end_ = spec_utils.mirroring(
                    self.data["high_end_process"], y_spec_m, input_high_end, self.mp
                )
                wav_instrument = spec_utils.cmb_spectrogram_to_wave(
                    y_spec_m, self.mp, input_high_end_h, input_high_end_
                )
            else:
                wav_instrument = spec_utils.cmb_spectrogram_to_wave(y_spec_m, self.mp)
            print("%s instruments done" % name)
            if model_params == "4band_v2":
                sf.write(
                    ins_file_path,
                    (np.array(wav_instrument) * 32768).astype("int16"), self.mp.param["sr"],
                )  #
            if model_params == "4band_v3":
                sf.write(
                    ins_file_path,
                    (np.array(wav_instrument) * 32768).astype("int16"), self.mp.param["sr"],
                )  #
        if vocal_file_path is not None and vocal_file_path != '':
            if self.data["high_end_process"].startswith("mirroring"):
                input_high_end_ = spec_utils.mirroring(
                    self.data["high_end_process"], v_spec_m, input_high_end, self.mp
                )
                wav_vocals = spec_utils.cmb_spectrogram_to_wave(
                    v_spec_m, self.mp, input_high_end_h, input_high_end_
                )
            else:
                wav_vocals = spec_utils.cmb_spectrogram_to_wave(v_spec_m, self.mp)
            print("%s vocals done" % name)
            if model_params == "4band_v2":
                sf.write(
                    vocal_file_path,
                    (np.array(wav_vocals) * 32768).astype("int16"), self.mp.param["sr"],
                )
            if model_params == "4band_v3":
                sf.write(
                    vocal_file_path,
                    (np.array(wav_vocals) * 32768).astype("int16"), self.mp.param["sr"],
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process audio with specified parameters.")
    parser.add_argument("-device", choices=["cpu", "cuda"], default="cpu", help="Device for processing")
    parser.add_argument("-is_half", default=False, type=bool, help="Use half precision")
    parser.add_argument("-model_path", default="uvr5_weights/2_HP-UVR.pth", help="Path to the model weights")
    parser.add_argument("-agg", type=int, default=10, help="Aggregation parameter")
    parser.add_argument("-audio_path", required=True, help="Path to the audio file")
    parser.add_argument("-output_vocal_path", required=True, help="Path to save the output vocal audio file")
    parser.add_argument("-output_background_path", default='', help="Path to save the output background audio file")
    parser.add_argument("-model_params", default="4band_v2", choices=["4band_v3", "4band_v2"], help="model params")
    parser.add_argument("-format", choices=["wav", "flac"], default="wav",)
    args = parser.parse_args()
    separator = AudioSeparator(
        model_path=args.model_path,
        device=args.device,
        is_half=args.is_half,
        agg=args.agg,
        model_params=args.model_params
    )
    separator.separate(
        args.audio_path,
        args.output_vocal_path,
        args.output_background_path,
        args.model_params,
        args.format
    )
