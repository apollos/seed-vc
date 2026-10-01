import os
from pathlib import Path

from huggingface_hub import hf_hub_download


_LOCAL_OVERRIDES = {
    ("funasr/campplus", "campplus_cn_common.bin"): "SEED_VC_CAMPPLUS_PATH",
    ("FunAudioLLM/CosyVoice-300M", "hift.pt"): "SEED_VC_HIFT_PATH",
}


def _model_file(repo_id, filename):
    env_name = _LOCAL_OVERRIDES.get((repo_id, filename))
    if env_name and env_name in os.environ:
        path = Path(os.environ[env_name]).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"{env_name} 指向的文件不存在：{path}")
        return str(path)

    os.makedirs("./checkpoints", exist_ok=True)
    return hf_hub_download(
        repo_id=repo_id, filename=filename, cache_dir="./checkpoints"
    )


def load_custom_model_from_hf(
    repo_id, model_filename="pytorch_model.bin", config_filename=None
):
    model_path = _model_file(repo_id, model_filename)
    if config_filename is None:
        return model_path
    return model_path, _model_file(repo_id, config_filename)
