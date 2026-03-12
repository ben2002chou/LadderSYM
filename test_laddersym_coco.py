import os
import torch
from inference_error import InferenceHandler
from glob import glob
from tqdm import tqdm
import librosa
import hydra
import numpy as np
import json

from evaluate_errors_coco import evaluate_main


def _validate_eval_inputs(cfg):
    help_msg = (
        "Set dataset paths via env vars:\n"
        "  LADDERSYM_COCO_ROOT, LADDERSYM_COCO_SPLIT_JSON\n"
        "or pass Hydra overrides like:\n"
        "  dataset.test.root_dir=/path dataset.test.split_json_path=/path/to/split.json"
    )
    if cfg.dataset.test.root_dir is None or str(cfg.dataset.test.root_dir).strip() == "":
        raise ValueError(f"dataset.test.root_dir is empty.\n{help_msg}")
    if cfg.dataset.test.split_json_path is None or str(cfg.dataset.test.split_json_path).strip() == "":
        raise ValueError(f"dataset.test.split_json_path is empty.\n{help_msg}")
    if not os.path.isdir(cfg.dataset.test.root_dir):
        raise FileNotFoundError(
            f"dataset.test.root_dir does not exist or is not a directory: {cfg.dataset.test.root_dir}"
        )
    if not os.path.isfile(cfg.dataset.test.split_json_path):
        raise FileNotFoundError(
            "dataset.test.split_json_path does not exist or is not a file: "
            f"{cfg.dataset.test.split_json_path}"
        )
    if cfg.path is None or str(cfg.path).strip() == "":
        raise ValueError("path is empty. Provide a checkpoint path via path=/path/to/model.ckpt")
    if not os.path.isfile(cfg.path):
        raise FileNotFoundError(f"Checkpoint file does not exist: {cfg.path}")


def _resolve_coco_cache_path():
    env_override = os.environ.get("LADDERSYM_COCO_FILE_PATH_CACHE")
    if env_override:
        return env_override
    runtime_output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    return os.path.join(runtime_output_dir, "file_paths_cache.json")


def get_scores(
    model,
    mistakes_audio_dir=None,
    scores_audio_dir=None,
    prompt_dir=None,
    mel_norm=True,
    eval_dataset="CocoChorales",
    exp_tag_name="test_midis",
    ground_truth=None,
    verbose=True,
    contiguous_inference=False,
    batch_size=1,
    max_length=1024,
    output_json_path=None,
):
    handler = InferenceHandler(
        model=model,
        device=torch.device("cuda"),
        mel_norm=mel_norm,
        contiguous_inference=contiguous_inference,
    )

    def extract_track_id(path):
        parts = os.path.normpath(path).split(os.sep)
        for marker in ("mistake", "score"):
            if marker in parts:
                idx = parts.index(marker)
                if idx + 1 < len(parts):
                    return parts[idx + 1]
        # Fallback to parent folder if expected markers are missing.
        return os.path.basename(os.path.dirname(path))

    def func(fname):
        audio, _ = librosa.load(fname, sr=16000)
        print(f"audio_len in seconds: {len(audio)/16000}")
        return audio
    
    if verbose:
        print("Total mistake audio files:", len(mistakes_audio_dir))
        print("Total score audio files:", len(scores_audio_dir))


    print(f"batch_size: {batch_size}")
    
    # Use a placeholder for prompt_file if prompt_dir is None
    placeholder_prompt = [None] * len(mistakes_audio_dir) if prompt_dir is None else prompt_dir

    # Iterate with a unified loop
    for mistake_file, score_file, prompt_file in tqdm(zip(mistakes_audio_dir, scores_audio_dir, placeholder_prompt), total=len(mistakes_audio_dir)):
        # Process each file pair here
        # if any file is missing, skip the pair
        if not os.path.exists(mistake_file) or not os.path.exists(score_file):
            print("Missing file(s) for:", mistake_file, score_file)
            continue
        print("Processing:", mistake_file, "and", score_file)
        mistake_audio = func(mistake_file)
        score_audio = func(score_file)

        fname = extract_track_id(mistake_file)
        base_name, ext = os.path.splitext(os.path.basename(mistake_file))
        file_name = base_name + ".mid"
        print(f"fname: {fname}, file_name: {file_name}", flush=True)
        outpath = os.path.join(exp_tag_name, fname, file_name)


        handler.inference(
            mistake_audio=mistake_audio,
            score_audio=score_audio,
            audio_path=fname,
            prompt_path=prompt_file,
            outpath=outpath,
            batch_size=batch_size,
            max_length=max_length,
            verbose=verbose,
        )

    if verbose:
        print("Evaluating...")
    current_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    # Evaluating the main function and getting the scores
    mean_scores, mean_track_instrument_scores, track_instrument_scores = evaluate_main(
        dataset_name=eval_dataset,
        test_midi_dir=os.path.join(current_dir, exp_tag_name),
        ground_truth=ground_truth,
        output_json_file=output_json_path,
    )
    
    # Print mean_scores if verbose
    if verbose:
        print("Mean Scores:")
        for key in sorted(mean_scores):
            print("{}: {:.4}".format(key, mean_scores[key]))

        print("\nMean Track and Instrument Scores:")
        for track_index, instruments in sorted(mean_track_instrument_scores.items()):
            print("Track {}: ".format(track_index))
            for instrument, metrics in instruments.items():
                print("  Instrument: {}".format(instrument))
                for metric, score in metrics.items():
                    print("    {}: {:.4}".format(metric, score))
                    

    return mean_scores, mean_track_instrument_scores, track_instrument_scores

def _load_MAESTRO_split_info(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
        midi_filename_to_number = {
            os.path.basename(path).replace(".midi", ""): str(number)
            for number, path in data["midi_filename"].items()
        }
        split_to_numbers = {split: set() for split in set(data["split"].values())}
        for number, split in data["split"].items():
            split_to_numbers[split].add(str(number))
        return midi_filename_to_number, split_to_numbers
    
def capitalize_instrument_name(file_path):
    # Extract the base path and file name
    base_path, file_name = os.path.split(file_path)
    # Split the file name into parts before and after the first underscore
    parts = file_name.split('_', 1)  # Only split on the first underscore
    if len(parts) > 1:
        # Split the instrument name part into individual words
        instrument_name_parts = parts[1].split('_')
        # Capitalize each part of the instrument name
        capitalized_instrument_name_parts = [part.capitalize() for part in instrument_name_parts]
        # Join the capitalized parts back with spaces
        capitalized_instrument_name = ' '.join(capitalized_instrument_name_parts)
        # Replace the original instrument part with the capitalized version
        parts[1] = capitalized_instrument_name
        # Reconstruct the file name
        file_name = '_'.join(parts) #.replace('_', ' ', 1)  # Replace only the first underscore with a space
        # Ensure only one .wav extension
        file_name = file_name.replace('.wav.wav', '.wav')
    # Reconstruct the full path
    return os.path.join(base_path, file_name)


def _build_dataset(root_dir, json_path, split, output_json_file):
    # Load the mapping and splits
    midi_filename_to_number, split_to_numbers = _load_MAESTRO_split_info(json_path)
    print(f"Finished loading {json_path}", flush=True)
    desired_file_numbers = split_to_numbers[split]

    df = []
    mistakes_audio_dir = []
    scores_audio_dir = []

    # Patterns for file discovery
    extra_notes_dir = os.path.join(root_dir, "label", "extra_notes")
    removed_notes_dir = os.path.join(root_dir, "label", "removed_notes")
    correct_notes_dir = os.path.join(root_dir, "label", "correct_notes")
    mistake_dir = os.path.join(root_dir, "mistake")
    score_dir = os.path.join(root_dir, "score")

    print(f"Finished loading {root_dir}", flush=True)

    # Define directory mapping
    directories = {
        "extra_notes": extra_notes_dir,
        "removed_notes": removed_notes_dir,
        "correct_notes": correct_notes_dir,
        "mistake": mistake_dir,
        "score": score_dir
    }

    def scan_and_save_paths(directories, output_json_file, batch_size=1000):
        files_dict = {key: {} for key in directories}
        batch = []
        batch_count = 0
        total_files = 0
        for key, dir_path in directories.items():
            for root, _, filenames in os.walk(dir_path):
                for filename in filenames:
                    if filename.endswith(".mid") or filename.endswith(".wav"):
                        file_path = os.path.join(root, filename)

                        if key in ["mistake", "score"]:
                            # Use stem MIDI files so index alignment matches extra/removed/correct labels.
                            if f"{os.sep}stems_midi{os.sep}" not in file_path or not file_path.endswith(".mid"):
                                continue

                        rel_parts = os.path.relpath(file_path, dir_path).split(os.sep)
                        if not rel_parts:
                            continue
                        dir_key = rel_parts[0]
                        if dir_key not in files_dict[key]:
                            files_dict[key][dir_key] = []
                        files_dict[key][dir_key].append(file_path)
                        batch.append(file_path)

                    if len(batch) >= batch_size:
                        batch_count += 1
                        total_files += len(batch)
                        print(f"Processed batch {batch_count} with {len(batch)} files (Total: {total_files})", flush=True)
                        batch.clear()

        if batch:  # Process remaining files in the last batch
            batch_count += 1
            total_files += len(batch)
            print(f"Processed batch {batch_count} with {len(batch)} files (Total: {total_files})", flush=True)

        with open(output_json_file, 'w') as json_file:
            json.dump(files_dict, json_file)
        print(f"File paths saved to {output_json_file}", flush=True)
        return files_dict

    def load_paths_from_json(output_json_file):
        with open(output_json_file, 'r') as json_file:
            files_dict = json.load(json_file)
        print(f"File paths loaded from {output_json_file}", flush=True)
        return files_dict

    def cache_has_stem_midi(files_dict, key):
        return any(
            f"{os.sep}stems_midi{os.sep}" in path and path.endswith(".mid")
            for paths in files_dict.get(key, {}).values()
            for path in paths
        )

    def cache_track_ids_are_aligned(files_dict):
        extra_track_ids = set(files_dict.get("extra_notes", {}).keys())
        if not extra_track_ids:
            return False
        mistake_track_ids = set(files_dict.get("mistake", {}).keys())
        score_track_ids = set(files_dict.get("score", {}).keys())
        overlap = extra_track_ids & mistake_track_ids & score_track_ids
        if not overlap:
            return False
        # Spot-check one overlapping track for stem MIDI availability.
        sample_track = next(iter(overlap))
        mistake_paths = files_dict.get("mistake", {}).get(sample_track, [])
        score_paths = files_dict.get("score", {}).get(sample_track, [])
        has_mistake_stem = any(
            f"{os.sep}stems_midi{os.sep}" in p and p.endswith(".mid") for p in mistake_paths
        )
        has_score_stem = any(
            f"{os.sep}stems_midi{os.sep}" in p and p.endswith(".mid") for p in score_paths
        )
        return has_mistake_stem and has_score_stem

    # Load or scan file paths
    if os.path.exists(output_json_file):
        print(f"Loading file paths from {output_json_file}", flush=True)
        files_dict = load_paths_from_json(output_json_file)
        # Older caches may include mix files for mistake/score and break stem alignment.
        if not (
            cache_has_stem_midi(files_dict, "mistake")
            and cache_has_stem_midi(files_dict, "score")
            and cache_track_ids_are_aligned(files_dict)
        ):
            print("Cached file paths are stale; rescanning dataset for stem MIDI alignment.", flush=True)
            files_dict = scan_and_save_paths(directories, output_json_file)
    else:
        print(f"Scanning and saving file paths to {output_json_file}", flush=True)
        files_dict = scan_and_save_paths(directories, output_json_file)

    extra_notes_files = files_dict["extra_notes"]
    removed_notes_files = files_dict["removed_notes"]
    correct_notes_files = files_dict["correct_notes"]
    mistake_files = files_dict["mistake"]
    score_files = files_dict["score"]

    # Ensure deterministic pairing by sorting discovered paths.
    for file_map in [extra_notes_files, removed_notes_files, correct_notes_files, mistake_files, score_files]:
        for track_id in file_map:
            file_map[track_id] = sorted(file_map[track_id])

    # Match files based on the common identifier
    for track_id in extra_notes_files.keys():
        file_number = midi_filename_to_number.get(track_id)
        if file_number in desired_file_numbers:
            num_subtracks = len(extra_notes_files[track_id])
            if track_id in removed_notes_files and track_id in correct_notes_files and track_id in mistake_files and track_id in score_files:
                for i in range(num_subtracks):
                    if (i < len(removed_notes_files[track_id]) and
                        i < len(correct_notes_files[track_id]) and
                        i < len(mistake_files[track_id]) and
                        i < len(score_files[track_id])):
                        mistake_audio = capitalize_instrument_name(mistake_files[track_id][i].replace("stems_midi", "stems_audio").replace(".mid", ".wav"))
                        score_audio = capitalize_instrument_name(score_files[track_id][i].replace("stems_midi", "stems_audio").replace(".mid", ".wav"))
                        
                        if os.path.exists(mistake_audio) and os.path.exists(score_audio):
                            df.append({
                                "track_id": track_id,
                                "file_number": file_number,
                                "extra_notes_midi": extra_notes_files[track_id][i],
                                "removed_notes_midi": removed_notes_files[track_id][i],
                                "correct_notes_midi": correct_notes_files[track_id][i],
                                "mistake_audio": mistake_audio,
                                "score_audio": score_audio,
                                "prompt": score_files[track_id][i].replace(".wav", ".mid"),
                                "score_midi": score_files[track_id][i].replace("stems_audio", "stems_midi").replace(".wav", ".mid"),
                                "aligned_midi": score_files[track_id][i].replace("stems_audio", "stems_midi").replace(".wav", "_aligned.mid"),
                            })
                            mistakes_audio_dir.append(mistake_audio)
                            scores_audio_dir.append(score_audio)
                        else:
                            if not os.path.exists(mistake_audio):
                                print(f"File does not exist: {mistake_audio}")
                            if not os.path.exists(score_audio):
                                print(f"File does not exist: {score_audio}")
                    else:
                        print(f"Index out of range for track_id {track_id}, subtrack index {i}")
            else:
                pass
                # print(f"Missing track data for {track_id}")
        else:
            pass
            # print(f"Track {track_id} not in desired file numbers")

    assert len(df) > 0, "No matching files found. Check the dataset directory."
    print("Total files:", len(df))
    return df, mistakes_audio_dir, scores_audio_dir
    


@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(cfg):
    _validate_eval_inputs(cfg)
    assert (
        cfg.path.endswith(".pt")
        or cfg.path.endswith("pth")
        or cfg.path.endswith("ckpt")
    ), "Only .pt, .pth, .ckpt files are supported."
    assert cfg.eval.exp_tag_name
    output_json_path = _resolve_coco_cache_path()
    output_json_dir = os.path.dirname(output_json_path)
    if output_json_dir:
        os.makedirs(output_json_dir, exist_ok=True)
    print(f"Using Coco path cache: {output_json_path}", flush=True)
    dataset, mistakes_audio_dir, scores_audio_dir = _build_dataset(
        root_dir=cfg.dataset.test.root_dir,
        json_path=cfg.dataset.test.split_json_path,
        split="test",
        output_json_file=output_json_path,
    )
    cfg.model.config.use_prompt = cfg.use_prompt
    if cfg.model.config.use_prompt:
        # Extract ground truth prompts
        prompt_dir = [entry["prompt"] for entry in dataset]

    pl = hydra.utils.instantiate(cfg.model, optim_cfg=cfg.optim)
    print(f"Loading weights from: {cfg.path}")
    if cfg.path.endswith(".ckpt"):
        # load lightning module from checkpoint
        model_cls = hydra.utils.get_class(cfg.model._target_)
        print("torch.cuda.device_count():", torch.cuda.device_count())

        pl = model_cls.load_from_checkpoint(
            cfg.path,
            config=cfg.model.config,
            optim_cfg=cfg.optim,
        )
        model = pl.model
        model = torch.compile(model)
    else:
        # load weights for nn.Module
        model = pl.model
        model = torch.compile(model)
        if cfg.eval.load_weights_strict is not None:
            model.load_state_dict(
                torch.load(cfg.path), strict=cfg.eval.load_weights_strict
            )
        else:
            model.load_state_dict(torch.load(cfg.path), strict=False)

    model.eval()
    # if torch.cuda.is_available():
    #     model.cuda()

    
    
    if cfg.eval.eval_first_n_examples:
        random_offset = 0
        if len(mistakes_audio_dir) > cfg.eval.eval_first_n_examples:
            random_offset = np.random.randint(0, len(mistakes_audio_dir) - cfg.eval.eval_first_n_examples)
        else:
            cfg.eval.eval_first_n_examples = len(mistakes_audio_dir)
            
        mistakes_audio_dir = mistakes_audio_dir[random_offset: random_offset + cfg.eval.eval_first_n_examples]
        scores_audio_dir = scores_audio_dir[random_offset: random_offset + cfg.eval.eval_first_n_examples]
        if cfg.model.config.use_prompt:
            prompt_dir = prompt_dir[random_offset: random_offset + cfg.eval.eval_first_n_examples]  # Subset ground truth as well
        
        # mistakes_audio_dir = mistakes_audio_dir[: cfg.eval.eval_first_n_examples]
        # scores_audio_dir = scores_audio_dir[: cfg.eval.eval_first_n_examples]
        

    mel_norm = True
    

    get_scores(
        model,
        mistakes_audio_dir=mistakes_audio_dir,
        scores_audio_dir=scores_audio_dir,
        mel_norm=mel_norm,
        eval_dataset=cfg.eval.eval_dataset,
        exp_tag_name=cfg.eval.exp_tag_name,
        ground_truth=dataset,
        prompt_dir=prompt_dir if cfg.model.config.use_prompt else None,
        contiguous_inference=cfg.eval.contiguous_inference,
        batch_size=cfg.eval.batch_size,
        output_json_path=output_json_path,
    )


if __name__ == "__main__":
    main()
