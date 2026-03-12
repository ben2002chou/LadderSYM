import os
from typing import Iterable, Optional

import hydra
import librosa
import torch
from omegaconf import DictConfig

# Ensure PATH exists so downstream imports (e.g., pydub via note_seq) can locate
# system binaries in sanitized environments.
os.environ.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")

from inference_error import InferenceHandler

def load_audio(file_path, sr=16000):
    audio, _ = librosa.load(file_path, sr=sr)
    print(f"Loaded {file_path} with length {len(audio)/sr} seconds.")
    return audio

def run_inference(
    handler: InferenceHandler,
    mistake_file: str,
    score_file: str,
    prompt_file: Optional[str],
    output_dir: str,
    *,
    batch_size: int = 1,
    max_length: int = 1024,
    output_mid_name: Optional[str] = None,
    overwrite: bool = False,
):
    mistake_audio = load_audio(mistake_file)
    score_audio = load_audio(score_file)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(mistake_file))[0]
    if output_mid_name is None:
        output_mid_name = f"{base_name}_output.mid"

    piece_name = os.path.basename(os.path.normpath(output_dir)) or base_name
    audio_path_stem = os.path.splitext(output_mid_name)[0] or piece_name

    outpath = os.path.join(output_dir, output_mid_name)

    if os.path.exists(outpath) and not overwrite:
        print(f"Output {outpath} already exists. Skipping inference.")
        return

    # Attempt to locate the companion mistake MIDI so the handler can copy
    # its tempo map onto the generated output.
    mistake_midi_path = None
    candidate_mid = os.path.join(os.path.dirname(mistake_file), "mistake.mid")
    if os.path.exists(candidate_mid):
        mistake_midi_path = candidate_mid

    handler.inference(
        mistake_audio=mistake_audio,
        score_audio=score_audio,
        audio_path=audio_path_stem,
        prompt_path=prompt_file,
        outpath=outpath,
        mistake_midi_path=mistake_midi_path,
        batch_size=batch_size,
        max_length=max_length,
        verbose=True,
    )
    print(f"Inference completed. Output saved to {outpath}")


def resolve_model_label(cfg: DictConfig) -> str:
    """Derive a string label for naming inference outputs."""

    if cfg.get("model_label"):
        return cfg.model_label

    target = cfg.model.get("_target_") if cfg.get("model") else None
    if target:
        return target.split(".")[-1]

    return "model"


def iter_piece_directories(dataset_dir: str) -> Iterable[str]:
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    for entry in sorted(os.scandir(dataset_dir), key=lambda e: e.name):
        if entry.is_dir():
            yield entry.path


def collect_piece_paths(piece_dir: str, score_audio_name: str, mistake_audio_name: str, prompt_name: Optional[str]) -> Optional[dict]:
    score_audio_path = os.path.join(piece_dir, score_audio_name)
    mistake_audio_path = os.path.join(piece_dir, mistake_audio_name)
    prompt_path = os.path.join(piece_dir, prompt_name) if prompt_name else None

    missing = []
    if not os.path.exists(score_audio_path):
        missing.append(score_audio_path)
    if not os.path.exists(mistake_audio_path):
        missing.append(mistake_audio_path)
    if prompt_path and not os.path.exists(prompt_path):
        missing.append(prompt_path)

    if missing:
        print(f"Skipping {piece_dir}; missing files: {missing}")
        return None

    return {
        "score_audio": score_audio_path,
        "mistake_audio": mistake_audio_path,
        "prompt": prompt_path,
    }

@hydra.main(config_path=None, config_name=None, version_base="1.1")
def main(cfg: DictConfig):
    assert cfg.path, "Model path must be specified in the config file"
    print(torch.__version__, flush=True)
    print("cfg.model:",cfg.model, flush=True)
    cfg.model.config.use_prompt = cfg.use_prompt
    # Load the model using Hydra config
    pl = hydra.utils.instantiate(cfg.model, optim_cfg=cfg.optim)
    print(f"Loading weights from: {cfg.path}")
    
    if cfg.path.endswith(".ckpt"):
        # Load from a PyTorch Lightning checkpoint
        model_cls = hydra.utils.get_class(cfg.model._target_)
        pl = model_cls.load_from_checkpoint(cfg.path, config=cfg.model.config, optim_cfg=cfg.optim)
        model = pl.model
        
        # Check if the model's state_dict is loaded correctly
        state_dict = pl.state_dict()
        print("Loaded state dict keys:", state_dict.keys())
    else:
        print("Only .ckpt file loading is supported in this script.")
    
    model.eval()
    
    if torch.cuda.is_available():
        model.cuda()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    handler = InferenceHandler(model=model, device=device)

    model_label = resolve_model_label(cfg)
    output_mid_name = cfg.get("output_mid_name") or f"{model_label}_output.mid"
    overwrite = bool(cfg.get("overwrite", False))

    dataset_dir = cfg.get("dataset_dir")
    if dataset_dir:
        dataset_dir = os.path.abspath(os.path.expanduser(dataset_dir))
        score_audio_name = cfg.get("score_audio_filename", "score.wav")
        mistake_audio_name = cfg.get("mistake_audio_filename", "mistake.wav")
        prompt_name = cfg.get("prompt_filename", "score.mid")

        pieces = cfg.get("pieces")
        piece_filter = set(pieces) if pieces else None

        for piece_dir in iter_piece_directories(dataset_dir):
            piece_name = os.path.basename(piece_dir.rstrip(os.sep))
            if piece_filter and piece_name not in piece_filter:
                continue

            paths = collect_piece_paths(piece_dir, score_audio_name, mistake_audio_name, prompt_name)
            if not paths:
                continue

            print(f"Running inference for {piece_name}")
            run_inference(
                handler,
                paths["mistake_audio"],
                paths["score_audio"],
                paths["prompt"],
                piece_dir,
                batch_size=cfg.get("batch_size", 1),
                max_length=cfg.get("max_length", 1024),
                output_mid_name=output_mid_name,
                overwrite=overwrite,
            )
        return

    # Fallback single-example execution using explicit paths from config.
    mistake_file = cfg.get("mistake_file")
    score_file = cfg.get("score_file")
    output_dir = cfg.get("output_dir")
    prompt_file = cfg.get("prompt_file")

    if not all([mistake_file, score_file, output_dir]):
        raise ValueError(
            "Either dataset_dir must be provided or explicit mistake_file, "
            "score_file, and output_dir must be set."
        )

    run_inference(
        handler,
        mistake_file,
        score_file,
        prompt_file,
        output_dir,
        batch_size=cfg.get("batch_size", 1),
        max_length=cfg.get("max_length", 1024),
        output_mid_name=output_mid_name,
        overwrite=overwrite,
    )

if __name__ == "__main__":
    main()
