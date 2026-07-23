#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import onnxruntime as ort

# pyrefly: ignore [missing-import]
from diffsinger_utau.voice_bank import PredAcoustic, PredDuration, PredVariance, PredVocoder
# pyrefly: ignore [missing-import]
from diffsinger_utau.voice_bank.commons.ds_reader import DSReader
# pyrefly: ignore [missing-import]
from diffsinger_utau.voice_bank.commons.utils import resample_align_curve
# pyrefly: ignore [missing-import]
from diffsinger_utau.voice_bank.commons.voice_bank_reader import VoiceBankReader


def parse_args():
    parser = argparse.ArgumentParser(description="Render an 8-bar PMTM DiffSinger score.")
    parser.add_argument("--score", required=True)
    parser.add_argument("--voice-bank", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--lang", default="ko")
    parser.add_argument("--acoustic-steps", type=int, default=30)
    parser.add_argument("--variance-steps", type=int, default=20)
    parser.add_argument("--use-ai-dur", action="store_true", default=True, help="Enable hybrid AI duration scaling")
    parser.add_argument("--no-ai-dur", action="store_false", dest="use_ai_dur", help="Disable hybrid AI duration scaling")
    return parser.parse_args()


def select_device(requested):
    providers = set(ort.get_available_providers())
    if requested == "auto":
        if "CUDAExecutionProvider" in providers:
            return "cuda"
        if "CoreMLExecutionProvider" in providers:
            return "mps"
        return "cpu"
    if requested == "cuda" and "CUDAExecutionProvider" not in providers:
        raise RuntimeError("CUDAExecutionProvider is not available in this DiffSinger runtime.")
    if requested == "mps" and "CoreMLExecutionProvider" not in providers:
        raise RuntimeError("CoreMLExecutionProvider is not available in this DiffSinger runtime.")
    return requested


def render(score_path, voice_bank_path, output_path, device, lang, acoustic_steps, variance_steps, use_ai_dur=True):
    sections = DSReader(score_path).read_ds()
    if len(sections) not in (8, 16):
        raise RuntimeError(f"PMTM DiffSinger score must contain 8 or 16 sections. (currently: {len(sections)})")

    reader = VoiceBankReader(voice_bank_path)
    acoustic = PredAcoustic(reader.get_dsacoustic())
    variance = PredVariance(reader.get_dsvariance())
    vocoder = PredVocoder(reader.get_dsvocoder())
    sample_rate = vocoder.ds_vocoder.sample_rate
    
    dur_model = None
    if use_ai_dur and (voice_bank_path / "dsdur").is_dir():
        try:
            dur_model = PredDuration(reader.get_dsdur())
        except Exception as exc:
            print(f"Warning: Could not initialize PredDuration: {exc}", file=sys.stderr)

    rendered = []

    for index, section in enumerate(sections):
        section["lang"] = lang
        
        if dur_model is not None:
            try:
                pred_dur = dur_model.predict(section, lang=lang)
                if pred_dur is not None and len(pred_dur) > 0:
                    ph_dur_orig = [float(v) for v in section["ph_dur"].split()]
                    ph_num = [int(v) for v in section["ph_num"].split()]
                    
                    if len(ph_dur_orig) == len(pred_dur) and sum(ph_num) == len(ph_dur_orig):
                        scaled_dur = []
                        cursor = 0
                        for num in ph_num:
                            syllable_orig_dur = sum(ph_dur_orig[cursor : cursor + num])
                            ai_durs = [max(0.001, float(pred_dur[cursor + k])) for k in range(num)]
                            ai_sum = sum(ai_durs)
                            
                            if ai_sum > 0:
                                for k in range(num):
                                    scaled_dur.append(syllable_orig_dur * (ai_durs[k] / ai_sum))
                            else:
                                scaled_dur.extend(ph_dur_orig[cursor : cursor + num])
                            cursor += num
                            
                        diff = sum(ph_dur_orig) - sum(scaled_dur)
                        scaled_dur[-1] += diff
                        section["ph_dur"] = " ".join(f"{v:.6f}" for v in scaled_dur)
            except Exception as exc:
                print(f"Warning: Hybrid AI duration scaling failed for bar {index + 1}: {exc}", file=sys.stderr)

        predicted_variances = variance.predict(
            section,
            lang=lang,
            steps=variance_steps,
            retake_all=True,
        )
        output_names = [
            output.name[:-5] if output.name.endswith("_pred") else output.name
            for output in variance.dsvariance.variance_model.session.get_outputs()
        ]
        output_values = list(predicted_variances.values())
        if len(output_names) != len(output_values):
            raise RuntimeError("DiffSinger variance output count does not match its model metadata.")
        for name, values in zip(output_names, output_values):
            section[name] = " ".join(
                str(float(value)) for value in np.clip(values, -96.0, 0.0)
            )
            section[f"{name}_timestep"] = str(variance.timestep)

        mel = acoustic.predict(
            section,
            lang=lang,
            steps=acoustic_steps,
            device=device,
        )
        f0 = resample_align_curve(
            np.array(section["f0_seq"].split(), dtype=np.float32),
            original_timestep=float(section["f0_timestep"]),
            target_timestep=vocoder.timestep,
            align_length=mel.shape[1],
        )
        wav = vocoder.predict(mel, f0, device=device).astype(np.float32)
        rendered.append((float(section["offset"]), wav))
        print("Rendered bar {}/{}".format(index + 1, len(sections)), flush=True)


    total_samples = max(int(round(offset * sample_rate)) + len(wav) for offset, wav in rendered)
    mixed = np.zeros(total_samples, dtype=np.float32)
    for offset, wav in rendered:
        start = int(round(offset * sample_rate))
        mixed[start : start + len(wav)] += wav
    mixed = np.clip(mixed, -1.0, 1.0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    vocoder.save_wav(mixed, output_path)


def main():
    args = parse_args()
    try:
        render(
            Path(args.score),
            Path(args.voice_bank),
            Path(args.output),
            select_device(args.device),
            args.lang,
            args.acoustic_steps,
            args.variance_steps,
            args.use_ai_dur,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
