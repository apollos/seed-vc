#!/usr/bin/env python3
"""Pseudo-realtime microphone voice conversion with Seed-VC.

This runner keeps Seed-VC and the reference-speaker features resident in
memory.  Microphone audio is processed in overlapping windows and sent to the
system default playback device.  It is intended as a latency/quality test;
Seed-VC itself is not a native low-latency streaming model.
"""

from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_CHECKPOINT = Path(
    "~/local-model-workbench/models/Plachta/Seed-VC/tiny/DiT_uvit_tat_xlsr_ema.pth"
).expanduser()
DEFAULT_CONFIG = Path(
    "~/local-model-workbench/models/Plachta/Seed-VC/tiny/"
    "config_dit_mel_seed_uvit_xlsr_tiny_local.yml"
).expanduser()
DEFAULT_REFERENCE = PROJECT_ROOT / "examples/reference/s3p2.wav"
DEFAULT_CAMPPLUS = Path(
    "~/local-model-workbench/models/IndexTeam/IndexTTS-2.5/"
    "hf_cache/campplus_cn_common.bin"
).expanduser()
DEFAULT_HIFT = Path(
    "~/local-model-workbench/models/FunAudioLLM/CosyVoice-300M/hift.pt"
).expanduser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed-VC microphone pseudo-realtime conversion test"
    )
    parser.add_argument("--target", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--campplus", type=Path, default=DEFAULT_CAMPPLUS)
    parser.add_argument("--hift", type=Path, default=DEFAULT_HIFT)
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--cfg-rate", type=float, default=0.7)
    parser.add_argument("--chunk-seconds", type=float, default=2.0)
    parser.add_argument("--overlap-seconds", type=float, default=0.25)
    parser.add_argument(
        "--audio-blocksize",
        type=int,
        default=1024,
        help="PortAudio capture block size in samples",
    )
    parser.add_argument(
        "--silence-threshold",
        type=float,
        default=0.002,
        help="Skip conversion below this RMS level; use 0 to disable",
    )
    parser.add_argument(
        "--input-device",
        type=int,
        default=None,
        help="sounddevice input index; default follows the operating system",
    )
    parser.add_argument(
        "--output-device",
        type=int,
        default=None,
        help="sounddevice output index; default follows the operating system",
    )
    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def configure_offline_paths(args: argparse.Namespace) -> None:
    os.environ["SEED_VC_CAMPPLUS_PATH"] = str(require_file(args.campplus, "CampPlus"))
    os.environ["SEED_VC_HIFT_PATH"] = str(require_file(args.hift, "HiFT"))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


class SeedVCRealtimeEngine:
    def __init__(self, args: argparse.Namespace):
        # Seed-VC reads configs/hifigan.yml relative to the repository root.
        os.chdir(PROJECT_ROOT)
        sys.path.insert(0, str(PROJECT_ROOT))

        import librosa
        import torch
        import torchaudio

        import inference as seed_inference

        self.torch = torch
        self.torchaudio = torchaudio
        self.seed = seed_inference
        self.device = seed_inference.device
        self.diffusion_steps = args.diffusion_steps
        self.cfg_rate = args.cfg_rate

        load_args = SimpleNamespace(
            checkpoint=str(require_file(args.checkpoint, "Seed-VC checkpoint")),
            config=str(require_file(args.config, "Seed-VC config")),
            f0_condition=False,
            fp16=False,
        )

        print("[Init] Loading Seed-VC tiny model and auxiliary models...")
        (
            self.model,
            self.semantic_fn,
            self.f0_fn,
            self.vocoder_fn,
            self.campplus_model,
            self.mel_fn,
            self.mel_fn_args,
        ) = seed_inference.load_models(load_args)

        self.sr = int(self.mel_fn_args["sampling_rate"])
        self.hop_length = int(self.mel_fn_args["hop_size"])
        self.overlap_frame_len = 16
        self.overlap_wave_len = self.overlap_frame_len * self.hop_length

        target_path = require_file(args.target, "reference audio")
        print(f"[Init] Caching reference speaker: {target_path}")
        ref_audio = librosa.load(str(target_path), sr=self.sr, mono=True)[0]
        ref_audio = ref_audio[: self.sr * 25]
        if ref_audio.size < self.sr:
            raise ValueError("Reference audio must contain at least one second of audio")

        with torch.inference_mode():
            ref_tensor = torch.from_numpy(ref_audio).unsqueeze(0).float().to(self.device)
            ref_16k = torchaudio.functional.resample(ref_tensor, self.sr, 16000)
            ref_semantic = self.semantic_fn(ref_16k)
            self.ref_mel = self.mel_fn(ref_tensor.float())
            ref_lengths = torch.LongTensor([self.ref_mel.size(2)]).to(self.device)

            feat = torchaudio.compliance.kaldi.fbank(
                ref_16k,
                num_mel_bins=80,
                dither=0,
                sample_frequency=16000,
            )
            feat = feat - feat.mean(dim=0, keepdim=True)
            self.style = self.campplus_model(feat.unsqueeze(0))
            self.prompt_condition, *_ = self.model.length_regulator(
                ref_semantic,
                ylens=ref_lengths,
                n_quantizers=3,
                f0=None,
            )

        max_context_window = self.sr // self.hop_length * 30
        self.max_source_window = max_context_window - self.ref_mel.size(2)
        if self.max_source_window <= 0:
            raise ValueError("Reference audio is too long for the model context window")

        print(
            f"[Init] Ready: device={self.device}, sample_rate={self.sr}, "
            f"reference={ref_audio.size / self.sr:.2f}s"
        )

    def _autocast_context(self):
        # The validated tiny model uses float32 inference on ROCm.
        return nullcontext()

    @staticmethod
    def _fit_length(wave: np.ndarray, length: int) -> np.ndarray:
        wave = np.asarray(wave, dtype=np.float32).reshape(-1)
        if wave.size >= length:
            return wave[:length]
        return np.pad(wave, (0, length - wave.size))

    def convert(self, source_wave: np.ndarray) -> tuple[np.ndarray, float]:
        torch = self.torch
        start = time.perf_counter()

        source_wave = np.asarray(source_wave, dtype=np.float32).reshape(-1)
        source = torch.from_numpy(source_wave).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            source_16k = self.torchaudio.functional.resample(source, self.sr, 16000)
            source_semantic = self.semantic_fn(source_16k)
            source_mel = self.mel_fn(source.float())
            source_lengths = torch.LongTensor([source_mel.size(2)]).to(self.device)
            condition, *_ = self.model.length_regulator(
                source_semantic,
                ylens=source_lengths,
                n_quantizers=3,
                f0=None,
            )

            if condition.size(1) > self.max_source_window:
                raise ValueError(
                    "Microphone chunk exceeds the available Seed-VC context window"
                )

            cat_condition = torch.cat([self.prompt_condition, condition], dim=1)
            cat_lengths = torch.LongTensor([cat_condition.size(1)]).to(self.device)

            with self._autocast_context():
                converted_mel = self.model.cfm.inference(
                    cat_condition,
                    cat_lengths,
                    self.ref_mel,
                    self.style,
                    None,
                    self.diffusion_steps,
                    inference_cfg_rate=self.cfg_rate,
                )
                converted_mel = converted_mel[:, :, self.ref_mel.size(-1) :]

            converted = self.vocoder_fn(converted_mel.float()).squeeze()
            converted = converted.detach().float().cpu().numpy()

        elapsed = time.perf_counter() - start
        converted = self._fit_length(converted, source_wave.size)
        return np.clip(converted, -1.0, 1.0), elapsed


class RealtimeAudioLoop:
    def __init__(self, engine: SeedVCRealtimeEngine, args: argparse.Namespace):
        import sounddevice as sd

        self.sd = sd
        self.engine = engine
        self.args = args
        self.sr = engine.sr
        self.window_samples = int(round(args.chunk_seconds * self.sr))
        self.overlap_samples = int(round(args.overlap_seconds * self.sr))
        self.step_samples = self.window_samples - self.overlap_samples
        if self.window_samples <= 0 or self.step_samples <= 0:
            raise ValueError("chunk-seconds must be greater than overlap-seconds")

        # Keep only a few seconds of microphone data.  If processing ever falls
        # behind, the callback replaces the oldest block so latency cannot grow
        # without bound.
        queue_blocks = max(8, int(np.ceil(3.0 * self.sr / args.audio_blocksize)))
        self.input_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=queue_blocks)
        self.output_queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=8)
        self.stop_event = threading.Event()
        self.dropped_input_blocks = 0
        self.player_error: BaseException | None = None

    def _input_callback(self, indata, frames, time_info, status) -> None:
        if status:
            print(f"\n[Audio input] {status}", flush=True)
        try:
            self.input_queue.put_nowait(indata[:, 0].copy())
        except queue.Full:
            self.dropped_input_blocks += 1
            try:
                self.input_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.input_queue.put_nowait(indata[:, 0].copy())
            except queue.Full:
                pass

    def _player(self) -> None:
        try:
            with self.sd.OutputStream(
                samplerate=self.sr,
                channels=1,
                dtype="float32",
                device=self.args.output_device,
                latency="low",
            ) as stream:
                while True:
                    try:
                        wave = self.output_queue.get(timeout=0.1)
                    except queue.Empty:
                        if self.stop_event.is_set():
                            break
                        continue
                    if wave is None:
                        break
                    stream.write(np.asarray(wave, dtype=np.float32).reshape(-1, 1))
        except BaseException as exc:  # surface worker-thread failures in main loop
            self.player_error = exc
            self.stop_event.set()

    def _read_samples(self, pending: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
        while pending.size < count and not self.stop_event.is_set():
            if self.player_error is not None:
                raise RuntimeError("Audio playback failed") from self.player_error
            try:
                block = self.input_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            pending = np.concatenate((pending, block))
        return pending[:count], pending[count:]

    @staticmethod
    def _crossfade(previous_tail: np.ndarray, current_head: np.ndarray) -> np.ndarray:
        length = min(previous_tail.size, current_head.size)
        if length == 0:
            return np.empty(0, dtype=np.float32)
        phase = np.linspace(0.0, np.pi / 2.0, length, dtype=np.float32)
        fade_out = np.cos(phase) ** 2
        fade_in = np.sin(phase) ** 2
        return previous_tail[:length] * fade_out + current_head[:length] * fade_in

    def run(self) -> None:
        input_info = self.sd.query_devices(self.args.input_device, "input")
        output_info = self.sd.query_devices(self.args.output_device, "output")
        print(f"[Audio] Input:  {input_info['name']}")
        print(f"[Audio] Output: {output_info['name']}")
        print(
            f"[Stream] window={self.args.chunk_seconds:.2f}s, "
            f"overlap={self.args.overlap_seconds:.2f}s, "
            f"steps={self.args.diffusion_steps}"
        )

        # Trigger one full pass before opening the microphone.  The first ROCm
        # pass builds kernels/caches and is much slower than steady-state
        # inference; doing it here prevents a large stale-audio backlog.
        print("[Warmup] Running one discarded Seed-VC pass before capture...")
        warmup_wave = np.zeros(self.window_samples, dtype=np.float32)
        _, warmup_elapsed = self.engine.convert(warmup_wave)
        print(f"[Warmup] Complete in {warmup_elapsed:.3f}s")
        print("[Stream] Running. Speak into the microphone; press Ctrl+C to stop.")

        player = threading.Thread(target=self._player, name="seedvc-player", daemon=True)
        player.start()

        pending = np.empty(0, dtype=np.float32)
        previous_input_tail: np.ndarray | None = None
        previous_output_tail: np.ndarray | None = None
        index = 0

        try:
            with self.sd.InputStream(
                samplerate=self.sr,
                channels=1,
                dtype="float32",
                device=self.args.input_device,
                callback=self._input_callback,
                blocksize=self.args.audio_blocksize,
                latency="low",
            ):
                while not self.stop_event.is_set():
                    needed = self.window_samples if previous_input_tail is None else self.step_samples
                    fresh, pending = self._read_samples(pending, needed)
                    if fresh.size != needed:
                        break

                    if previous_input_tail is None:
                        window = fresh
                    else:
                        window = np.concatenate((previous_input_tail, fresh))
                    previous_input_tail = window[-self.overlap_samples :].copy()

                    index += 1
                    rms = float(np.sqrt(np.mean(np.square(window), dtype=np.float64)))
                    if self.args.silence_threshold > 0 and rms < self.args.silence_threshold:
                        converted = np.zeros(self.window_samples, dtype=np.float32)
                        elapsed = 0.0
                    else:
                        converted, elapsed = self.engine.convert(window)

                    if previous_output_tail is None:
                        playable = converted[: -self.overlap_samples]
                    else:
                        blended = self._crossfade(
                            previous_output_tail,
                            converted[: self.overlap_samples],
                        )
                        middle = converted[self.overlap_samples : -self.overlap_samples]
                        playable = np.concatenate((blended, middle))
                    previous_output_tail = converted[-self.overlap_samples :].copy()

                    self.output_queue.put(playable)
                    rtf = elapsed / self.args.chunk_seconds
                    queued = self.input_queue.qsize()
                    print(
                        f"[Chunk {index:04d}] rms={rms:.4f} "
                        f"infer={elapsed:.3f}s RTF={rtf:.3f} "
                        f"input_queue={queued} dropped={self.dropped_input_blocks}",
                        flush=True,
                    )
        except KeyboardInterrupt:
            print("\n[Stream] Stopping...")
        finally:
            self.stop_event.set()
            try:
                self.output_queue.put_nowait(None)
            except queue.Full:
                pass
            player.join(timeout=3.0)
            if self.player_error is not None:
                raise RuntimeError("Audio playback failed") from self.player_error


def main() -> None:
    args = parse_args()
    configure_offline_paths(args)

    if args.diffusion_steps < 1:
        raise ValueError("diffusion-steps must be at least 1")
    if args.overlap_seconds <= 0:
        raise ValueError("overlap-seconds must be greater than zero")
    if args.audio_blocksize < 1:
        raise ValueError("audio-blocksize must be at least 1")

    engine = SeedVCRealtimeEngine(args)
    RealtimeAudioLoop(engine, args).run()


if __name__ == "__main__":
    main()
