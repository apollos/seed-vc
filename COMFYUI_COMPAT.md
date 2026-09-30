# ComfyUI / ROCm compatibility notes

This branch records the Seed-VC changes validated alongside the local ComfyUI
installation on 2026-09-30.  Seed-VC is imported and run in the same Python
environment as ComfyUI; it is not exposed through a separate service.

## Local auxiliary models

`hf_utils.py` supports explicit paths for the two auxiliary weights that would
otherwise be resolved through Hugging Face:

- `SEED_VC_CAMPPLUS_PATH` for `campplus_cn_common.bin`
- `SEED_VC_HIFT_PATH` for `hift.pt`

The normal Hugging Face download path remains available when an override is not
set.  `realtime_seedvc.py` validates both paths and enables offline mode before
loading Seed-VC.

## Pseudo-real-time test

`realtime_seedvc.py` keeps the tiny model and reference-speaker features in
memory, reads a microphone through `sounddevice`, converts overlapping windows,
and writes the result to the selected playback device.

```bash
python realtime_seedvc.py \
  --target examples/reference/s3p2.wav \
  --checkpoint "$HOME/local-model-workbench/models/Plachta/Seed-VC/tiny/DiT_uvit_tat_xlsr_ema.pth" \
  --config "$HOME/local-model-workbench/models/Plachta/Seed-VC/tiny/config_dit_mel_seed_uvit_xlsr_tiny_local.yml" \
  --campplus "$HOME/local-model-workbench/models/IndexTeam/IndexTTS-2.5/hf_cache/campplus_cn_common.bin" \
  --hift "$HOME/local-model-workbench/models/FunAudioLLM/CosyVoice-300M/hift.pt"
```

This is an overlapping-window latency/quality experiment, not native streaming.
Shorter chunks lower latency but can noticeably reduce voice quality.  The
Linux `default` or `pipewire` device is preferred because a raw ALSA `hw:*`
device may reject Seed-VC's 22.05 kHz output rate.

## Related ComfyUI node

The offline and real-time control nodes live in
[`apollos/local-comfyui-nodes`](https://github.com/apollos/local-comfyui-nodes/tree/main/ComfyUI-VoiceConversion).
Model checkpoints, reference audio, generated recordings, and virtual
environments are deliberately excluded from Git.
