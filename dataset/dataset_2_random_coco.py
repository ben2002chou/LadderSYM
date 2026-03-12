import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
import matplotlib.pyplot as plt
import numpy as np
import json
import random
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import librosa
import note_seq
from glob import glob
from contrib import (
    event_codec,
    note_sequences,
    spectrograms,
    vocabularies,
    run_length_encoding,
    metrics_utils,
)
from contrib.preprocessor import (
    class_to_error,
    add_track_to_notesequence,
    PitchBendError,
)

MIN_LOG_MEL = -12
MAX_LOG_MEL = 5


class Dataset(Dataset):

    def __init__(
        self,
        root_dir,
        split_json_path,
        split,
        mel_length=256,
        event_length=1024,
        prompt_length=1024,
        is_train=True,
        include_ties=True,
        ignore_pitch_bends=True,
        onsets_only=False,
        audio_filename="mix_16k.wav",
        midi_folder="MIDI",
        shuffle=True,
        num_rows_per_batch=8,
        split_frame_length=2000,
        is_randomize_tokens=True,
        is_random_alignment_shift_augmentation=False,
        use_prompt=True,
        is_deterministic=False,
    ) -> None:
        super().__init__()
        self.spectrogram_config = spectrograms.SpectrogramConfig()

        self.codec = vocabularies.build_codec(
            vocab_config=vocabularies.VocabularyConfig(num_velocity_bins=1)
        )
        self.vocab = vocabularies.vocabulary_from_codec(self.codec)
        self.audio_filename = audio_filename
        self.midi_folder = midi_folder
        self.mel_length = mel_length
        self.event_length = event_length
        self.prompt_length = prompt_length  # Initialize prompt_length
        # Cache discovered paths per dataset root to avoid cross-root reuse.
        root_cache_tag = re.sub(
            r"[^A-Za-z0-9._-]+", "_", os.path.basename(os.path.normpath(root_dir))
        )
        output_json_path = os.path.join(
            os.path.dirname(split_json_path), f"file_paths_{root_cache_tag}.json"
        )
        self.df = self._build_dataset(root_dir, split_json_path, split, output_json_path, shuffle=shuffle)
        self.is_train = is_train
        self.include_ties = include_ties
        self.ignore_pitch_bends = ignore_pitch_bends
        self.onsets_only = onsets_only
        self.tie_token = (
            self.codec.encode_event(event_codec.Event("tie", 0))
            if self.include_ties
            else None
        )
        self.num_rows_per_batch = num_rows_per_batch
        self.split_frame_length = split_frame_length
        self.is_deterministic = is_deterministic
        self.is_randomize_tokens = is_randomize_tokens
        self.random_alignment_shift_augmentation = is_random_alignment_shift_augmentation
        self.context_multiplier = 2
        self.use_prompt = use_prompt

    def _load_MAESTRO_split_info(self, json_path):
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



    def capitalize_instrument_name(self, file_path):
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
    
    def lowercase_instrument_name(self, file_path):
    
        # Extract the base path and file name
        base_path, file_name = os.path.split(file_path)
        
        # Split the file name into parts before and after the first underscore
        parts = file_name.split('_', 1)  # Only split on the first underscore
        if len(parts) > 1:
            # Split the instrument name part into individual words
            instrument_name_parts = parts[1].split('_')
            
            # Convert each part of the instrument name to lowercase
            lowercase_instrument_name_parts = [part.lower() for part in instrument_name_parts]
            
            # Join the lowercase parts back with underscores
            lowercase_instrument_name = '_'.join(lowercase_instrument_name_parts)
            
            # Replace the original instrument part with the lowercase version
            parts[1] = lowercase_instrument_name
            
            # Reconstruct the file name
            file_name = '_'.join(parts)
            
            # Ensure only one .wav extension
            file_name = file_name.replace('.wav.wav', '.wav').replace(' ', '_')
        
        # Reconstruct the full path
        return os.path.join(base_path, file_name)


    def _build_dataset(self, root_dir, json_path, split, output_json_file, shuffle=True):
        # Load the mapping and splits
        midi_filename_to_number, split_to_numbers = self._load_MAESTRO_split_info(json_path)
        print(f"Finished loading {json_path}", flush=True)
        desired_file_numbers = split_to_numbers[split]

        df = []
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
                # followlinks=True allows split-specific symlink views of the dataset.
                for root, _, filenames in os.walk(dir_path, followlinks=True):
                    for filename in filenames:
                        if key in ["extra_notes", "removed_notes", "correct_notes"]:
                            if filename.endswith(".mid"):
                                file_path = os.path.join(root, filename)
                                dir_key = os.path.normpath(file_path).split(os.sep)[-3]
                                if dir_key not in files_dict[key]:
                                    files_dict[key][dir_key] = []
                                files_dict[key][dir_key].append(file_path)
                                batch.append(file_path)
                        else:
                            if not filename.endswith("mix.mid") and not filename.endswith("mix.wav"):
                                if filename.endswith(".mid") or filename.endswith(".wav"):
                                    file_path = os.path.join(root, filename)
                                    dir_key = os.path.normpath(file_path).split(os.sep)[-3]
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

        def paths_match_root(files_dict, expected_root):
            """Validate cached paths belong to the requested dataset root."""
            expected_prefix = os.path.normpath(expected_root) + os.sep
            for key in directories:
                key_dict = files_dict.get(key, {})
                for paths in key_dict.values():
                    if not paths:
                        continue
                    sample_path = os.path.normpath(paths[0])
                    if not sample_path.startswith(expected_prefix):
                        print(
                            f"Cache mismatch for {key}: {sample_path} (expected prefix: {expected_prefix})",
                            flush=True,
                        )
                        return False
                    break
            return True

        # Load or scan file paths
        legacy_output_json_file = os.path.join(
            os.path.dirname(output_json_file), "file_paths_8_12.json"
        )
        files_dict = None

        if os.path.exists(output_json_file):
            candidate = load_paths_from_json(output_json_file)
            if paths_match_root(candidate, root_dir):
                files_dict = candidate
            else:
                print(
                    f"Cached paths in {output_json_file} do not match root_dir={root_dir}. Rescanning.",
                    flush=True,
                )

        if files_dict is None and os.path.exists(legacy_output_json_file):
            legacy_candidate = load_paths_from_json(legacy_output_json_file)
            if paths_match_root(legacy_candidate, root_dir):
                files_dict = legacy_candidate
                with open(output_json_file, "w") as json_file:
                    json.dump(files_dict, json_file)
                print(
                    f"Copied compatible legacy cache to {output_json_file}",
                    flush=True,
                )
            else:
                print(
                    f"Legacy cache {legacy_output_json_file} does not match root_dir={root_dir}.",
                    flush=True,
                )

        if files_dict is None:
            files_dict = scan_and_save_paths(directories, output_json_file)

        # Example usage: accessing file paths from the loaded dictionary
        extra_notes_files = files_dict["extra_notes"]
        removed_notes_files = files_dict["removed_notes"]
        correct_notes_files = files_dict["correct_notes"]
        mistake_files = files_dict["mistake"]
        score_files = files_dict["score"]

        
        # Match files based on the common identifier
        for track_id in extra_notes_files.keys():
            file_number = midi_filename_to_number.get(track_id)
            if file_number in desired_file_numbers:
                # Ensure all file types have the same number of subtracks
                num_subtracks = len(extra_notes_files[track_id])
                if track_id in removed_notes_files and track_id in correct_notes_files and track_id in mistake_files and track_id in score_files:
                    # Iterate over each subtrack index
                    for i in range(num_subtracks):
                        # Ensure that all data types have this subtrack index
                        if (i < len(removed_notes_files[track_id]) and i < len(correct_notes_files[track_id]) and
                            i < len(mistake_files[track_id]) and i < len(score_files[track_id])):
                            df.append({
                                "extra_notes_midi": extra_notes_files[track_id][i],
                                "removed_notes_midi": removed_notes_files[track_id][i],
                                "correct_notes_midi": correct_notes_files[track_id][i],
                                "mistake_audio": self.capitalize_instrument_name(mistake_files[track_id][i].replace("stems_midi", "stems_audio").replace(".mid", ".wav")),
                                "score_audio": self.capitalize_instrument_name(score_files[track_id][i].replace("stems_midi", "stems_audio").replace(".mid", ".wav"))
                            })

        assert len(df) > 0, "No matching files found. Check the dataset directory."
        print("Total files:", len(df), flush=True)
        
        # Shuffle the dataset if needed
        if shuffle:
            random.shuffle(df)

        return df

    def _audio_to_frames(
        self,
        samples: Sequence[float],
    ) -> Tuple[Sequence[Sequence[int]], np.ndarray]:
        """Convert audio samples to non-overlapping frames and frame times."""
        frame_size = self.spectrogram_config.hop_width
        samples = np.pad(
            samples, [0, frame_size - len(samples) % frame_size], mode="constant"
        )

        frames = spectrograms.split_audio(samples, self.spectrogram_config)

        num_frames = len(samples) // frame_size

        times = np.arange(num_frames) / self.spectrogram_config.frames_per_second
        return frames, times

    # def _parse_midi(self, path, instrument_dict: Dict[str, str]):
    #     """
    #     __getitem__ loads the MIDIfiles as a list of note sequences and a list of instrument values.
    #     Parses MIDI files and returns note sequences and instrument values.

    #     Args:
    #         path (str): The path to the directory containing the MIDI files.
    #         instrument_dict (Dict[str, str]): A dictionary mapping filenames to instrument values.

    #     Returns:
    #         Tuple[List[NoteSequence], List[str]]: A tuple containing a list of note sequences and a list of instrument values.
    #     """
    #     note_seqs = []

    #     for filename in instrument_dict.keys():
    #         # this can be pretty_midi.PrettyMIDI() obj / string path to midi
    #         midi_path = f"{path}/{filename}.mid"
    #         note_seqs.append(note_seq.midi_file_to_note_sequence(midi_path))
    #     return note_seqs, instrument_dict.values()

    def _tokenize(
        self,
        tracks: List[note_seq.NoteSequence],
        mistake_samples: np.ndarray,
        score_samples: np.ndarray,
        example_id: Optional[str] = None,  # not used
    ):

        frames_mistake, frame_times_mistake = self._audio_to_frames(mistake_samples)
        frames_score, frame_times_score = self._audio_to_frames(score_samples)

        # Add all the notes from the tracks to a single NoteSequence.
        ns = note_seq.NoteSequence(ticks_per_quarter=220)
        ns_score = note_seq.NoteSequence(ticks_per_quarter=220)
        # tracks = [note_seq.NoteSequence.FromString(seq) for seq in sequences]
        # print("tracks", len(tracks))
        for i, track in enumerate(tracks):
            # Mapping of error_classes
            if i == 3:
                error_class = 3  # Score has all correct notes.
                try:
                    add_track_to_notesequence(
                        ns_score,
                        track,
                        error_class=error_class,
                        ignore_pitch_bends=self.ignore_pitch_bends,
                    )
                except PitchBendError:
                    return
            else:
                error_class = i + 1
                try:
                    add_track_to_notesequence(
                        ns,
                        track,
                        error_class=error_class,
                        ignore_pitch_bends=self.ignore_pitch_bends,
                    )
                except PitchBendError:
                    return


        note_sequences.assign_error_classes(ns)
        note_sequences.validate_note_sequence(ns)
        if self.is_train:
            # Trim overlapping notes in training (as our event vocabulary cannot
            # represent them), but preserve original NoteSequence for eval.
            ns = note_sequences.trim_overlapping_notes(ns)

        if example_id is not None:
            ns.id = example_id

        if self.onsets_only:
            times, values = note_sequences.note_sequence_to_onsets(
                ns
            )  # get the start time of each note
        else:
            times, values = (
                note_sequences.note_sequence_to_onsets_and_offsets_and_programs(ns)
            )
            # print('times', times)
            # print('values', values)

        (
            events,
            event_start_indices,
            event_end_indices,
            state_events,
            state_event_indices,
        ) = run_length_encoding.encode_and_index_events(
            state=note_sequences.NoteEncodingState() if self.include_ties else None,
            event_times=times,
            event_values=values,
            encode_event_fn=note_sequences.note_event_data_to_events,
            codec=self.codec,
            frame_times=frame_times_mistake,
            encoding_state_to_events_fn=(
                note_sequences.note_encoding_state_to_events
                if self.include_ties
                else None
            ),
        )

        note_sequences.assign_error_classes(ns_score)
        note_sequences.validate_note_sequence(ns_score)
        if self.is_train:
            # Trim overlapping notes in training.
            ns_score = note_sequences.trim_overlapping_notes(ns_score)

        if example_id is not None:
            ns_score.id = example_id

        if self.onsets_only:
            times, values = note_sequences.note_sequence_to_onsets(ns_score)
        else:
            times, values = note_sequences.note_sequence_to_onsets_and_offsets_and_programs(ns_score)

        # check if times is empty
        # Encode events
        prompt_events, prompt_event_start_indices, prompt_event_end_indices, prompt_state_events, prompt_state_event_indices = run_length_encoding.encode_and_index_events(
            state=note_sequences.NoteEncodingState() if self.include_ties else None,
            event_times=times,
            event_values=values,
            encode_event_fn=note_sequences.note_event_data_to_events,
            codec=self.codec,
            frame_times=frame_times_score,
            encoding_state_to_events_fn=(
                note_sequences.note_encoding_state_to_events if self.include_ties else None
            ),
        )
        # # if length of prompt_event_start_indices is 0, then the prompt_events are probably empty, debug this.
        # if "prompt_event_start_indices" not in features or features["prompt_event_start_indices"].nelement() == 0:
        #     print("prompt_event_start_indices is empty", flush=True)
        #     print(f"prompt_events: {prompt_events}", flush=True)
        #     print(f"prompt_state_events: {prompt_state_events}", flush=True)
        #     print(f"prompt_state_event_indices: {prompt_state_event_indices}", flush=True)
        #     print(f"times: {times}", flush=True)
        #     print(f"values: {values}", flush=True)
        #     print(f"ns_score: {ns_score}", flush=True)


            


        return {
            "mistake_inputs": np.array(frames_mistake),
            "mistake_input_times": frame_times_mistake,
            "score_inputs": np.array(frames_score),
            "score_input_times": frame_times_score,
            "targets": events,
            "input_event_start_indices": event_start_indices,
            "input_event_end_indices": event_end_indices,
            "state_events": state_events,
            "input_state_event_indices": state_event_indices,
            "prompt_events": prompt_events,
            "prompt_event_start_indices": prompt_event_start_indices,
            "prompt_event_end_indices": prompt_event_end_indices,
            "prompt_state_events": prompt_state_events,
            "prompt_state_event_indices": prompt_state_event_indices,
        }

    def _extract_target_sequence_with_indices(
        self, features, state_events_end_token=None
    ):
        """Extract target sequence corresponding to audio token segment."""
        target_start_idx = features["input_event_start_indices"][0]
        target_end_idx = features["input_event_end_indices"][-1]

        features["targets"] = features["targets"][target_start_idx:target_end_idx]
        # print(
        #     "features[targets]", [self.get_token_name(k) for k in features["targets"]]
        # )

        if state_events_end_token is not None:
            # Extract the state events corresponding to the audio start token, and
            # prepend them to the targets array.
            state_event_start_idx = features["input_state_event_indices"][0]
            state_event_end_idx = state_event_start_idx + 1
            while (
                features["state_events"][state_event_end_idx - 1]
                != state_events_end_token
            ):
                state_event_end_idx += 1
                # Ensure we don't run off the end of the state events. # TODO: this is a hack!
                if state_event_end_idx >= len(features["state_events"]):
                    print("encountered end of state events without end token", flush=True)
                    break
                
            features["targets"] = np.concatenate(
                [
                    features["state_events"][state_event_start_idx:state_event_end_idx],
                    features["targets"],
                ],
                axis=0,
            )
            # print(
            #     "features[targets] 2",
            #     [self.get_token_name(k) for k in features["targets"]],
            # )

        """Extract prompt sequence corresponding to score audio token segment."""
        if features["prompt_event_start_indices"].nelement() == 0:
            print("Debug: 'prompt_event_start_indices' is empty or missing.", flush=True)
            print(f"Debug: 'prompt_events' state: {features.get('prompt_events', None)}", flush=True)

            features["prompt_events"] =  np.array([])  
            
        else:
            prompt_start_idx = features["prompt_event_start_indices"][0]
            prompt_end_idx = features["prompt_event_end_indices"][-1]
            
            features["prompt_events"] = features["prompt_events"][prompt_start_idx:prompt_end_idx]
        
        if state_events_end_token is not None:
            # Extract the state events corresponding to the prompt start token and
            # prepend them to the prompt_events array.
            if features["prompt_state_event_indices"].nelement() == 0:
                print("Debug: 'prompt_state_event_indices' is empty or missing.", flush=True)
                print(f"Debug: 'prompt_state_events' state: {features.get('prompt_state_events', None)}", flush=True)
                features["prompt_state_events"] = np.array([])
                features["prompt_events"] = np.concatenate(
                    [
                        features["prompt_state_events"],
                        features["prompt_events"],
                    ],
                    axis=0,
                )
            else: 
                prompt_state_event_start_idx = features["prompt_state_event_indices"][0]
                prompt_state_event_end_idx = prompt_state_event_start_idx + 1
                while (
                    features["prompt_state_events"][prompt_state_event_end_idx - 1]
                    != state_events_end_token
                ):
                    prompt_state_event_end_idx += 1
                    # Ensure we don't run off the end of the prompt_state_events.
                    if prompt_state_event_end_idx >= len(features["prompt_state_events"]):
                        print("Encountered end of prompt state events without end token", flush=True)
                        break
                    
                # print("features[prompt_state_events]", features["prompt_state_events"], flush=True)
                # print("prompt_state_event_start_idx", prompt_state_event_start_idx, flush=True)
                # print("prompt_state_event_end_idx", prompt_state_event_end_idx, flush=True)
                # print("features[prompt_events]", features["prompt_events"], flush=True)
                # print("prompt_start_idx", prompt_start_idx, flush=True)
                # print("prompt_end_idx", prompt_end_idx, flush=True)
                
                # check if targets have zeros start or end indices
                # if target_start_idx == 0:
                #     print("target_start_idx is 0", flush=True)
                #     print("features[input_event_start_indices]", features["input_event_start_indices"], flush=True)
                # if target_end_idx == 0:
                #     print("target_end_idx is 0", flush=True)
                #     print("features[input_event_end_indices]", features["input_event_end_indices"], flush=True)
                # if prompt_start_idx == 0:
                #     print("prompt_start_idx is 0", flush=True)
                #     print("features[prompt_event_start_indices]", features["prompt_event_start_indices"], flush=True)
                # if prompt_end_idx == 0:
                #     print("prompt_end_idx is 0", flush=True)
                #     print("features[prompt_event_end_indices]", features["prompt_event_end_indices"], flush=True)

                features["prompt_events"] = np.concatenate(
                    [
                        features["prompt_state_events"][
                            prompt_state_event_start_idx:prompt_state_event_end_idx
                        ],
                        features["prompt_events"],
                    ],
                    axis=0,
                )
                

        return features

    def run_length_encode_shifts(
        self,
        features,
        state_change_event_types=["velocity", "error_class"],
        feature_key="targets",
    ):
        state_change_event_ranges = [
            self.codec.event_type_range(event_type)
            for event_type in state_change_event_types
        ]

        events = features[feature_key]

        shift_steps = 0
        total_shift_steps = 0
        output = []

        current_state = np.zeros(len(state_change_event_ranges), dtype=np.int32)
        for j, event in enumerate(events):
            if self.codec.is_shift_event_index(event):
                shift_steps += 1
                total_shift_steps += 1
            else:
                # If this event is a state change and has the same value as the current
                # state, we can skip it entirely.

                # NOTE: this needs to be uncommented if not using random-order augmentation
                # because random-order augmentation use `_remove_redundant_tokens` to replace this part
                if not self.is_randomize_tokens or feature_key == "prompt_events":
                    is_redundant = False
                    for i, (min_index, max_index) in enumerate(
                        state_change_event_ranges
                    ):
                        if (min_index <= event) and (event <= max_index):
                            if current_state[i] == event:
                                is_redundant = True
                            current_state[i] = event
                    if is_redundant:
                        continue

                # Once we've reached a non-shift event, RLE all previous shift events
                # before outputting the non-shift event.
                if shift_steps > 0:
                    shift_steps = total_shift_steps
                    while shift_steps > 0:
                        output_steps = np.minimum(
                            self.codec.max_shift_steps, shift_steps
                        )
                        output = np.concatenate([output, [output_steps]], axis=0)
                        shift_steps -= output_steps
                output = np.concatenate([output, [event]], axis=0)

        features[feature_key] = output
        return features

    def remove_redundant_tokens(
        self,
        events,
        state_change_event_types=["velocity", "error_class"],
    ):
        state_change_event_ranges = [
            self.codec.event_type_range(event_type)
            for event_type in state_change_event_types
        ]

        output = []

        current_state = np.zeros(len(state_change_event_ranges), dtype=np.int32)
        for j, event in enumerate(events):
            # If this event is a state change and has the same value as the current
            # state, we can skip it entirely.

            is_redundant = False
            for i, (min_index, max_index) in enumerate(state_change_event_ranges):
                if (min_index <= event) and (event <= max_index):
                    if current_state[i] == event:
                        is_redundant = True
                    current_state[i] = event
            if is_redundant:
                continue

            # Once we've reached a non-shift event, RLE all previous shift events
            # before outputting the non-shift event.
            output = np.concatenate([output, [event]], axis=0)

        return output

    def _compute_spectrogram(self, ex):
        # Process 'mistake_inputs'
        mistake_samples = spectrograms.flatten_frames(ex["mistake_inputs"])
        if mistake_samples.shape[0] == 0:
            # assign a default empty waveform of 0's to the mistake samples
            # this is a temporary fix to handle the case where the mistake samples are empty
            mistake_samples = np.zeros(
                self.mel_length * self.spectrogram_config.hop_width
            )

        # Check if the length is less than required for the FFT
        if mistake_samples.shape[0] < self.spectrogram_config.n_fft:
            pad_width = self.spectrogram_config.n_fft - mistake_samples.shape[0]
            mistake_samples = np.pad(mistake_samples, (0, pad_width), mode='constant')


        # Compute spectrogram
        ex["mistake_inputs"] = torch.from_numpy(
            np.array(
                spectrograms.compute_spectrogram(
                    mistake_samples, self.spectrogram_config
                )
            )
        )

        # Normalize the spectrogram
        ex["mistake_inputs"] = torch.clamp(
            ex["mistake_inputs"], min=MIN_LOG_MEL, max=MAX_LOG_MEL
        )
        ex["mistake_inputs"] = (ex["mistake_inputs"] - MIN_LOG_MEL) / (
            MAX_LOG_MEL - MIN_LOG_MEL
        )

        # Process 'score_inputs' similarly
        score_samples = spectrograms.flatten_frames(ex["score_inputs"])
        if score_samples.shape[0] == 0:
            # assign a default empty waveform of 0's to the score samples
            # this is a temporary fix to handle the case where the score samples are empty
            score_samples = np.zeros(
                self.mel_length * self.spectrogram_config.hop_width
            )


        if score_samples.shape[0] < self.spectrogram_config.n_fft:
            pad_width = self.spectrogram_config.n_fft - score_samples.shape[0]
            score_samples = np.pad(score_samples, (0, pad_width), mode='constant')


        # Compute spectrogram
        ex["score_inputs"] = torch.from_numpy(
            np.array(
                spectrograms.compute_spectrogram(
                    score_samples,
                    self.spectrogram_config,
                    context_multiplier=self.context_multiplier,
                )
            )
        )

        # Normalize the spectrogram
        ex["score_inputs"] = torch.clamp(
            ex["score_inputs"], min=MIN_LOG_MEL, max=MAX_LOG_MEL
        )
        ex["score_inputs"] = (ex["score_inputs"] - MIN_LOG_MEL) / (
            MAX_LOG_MEL - MIN_LOG_MEL
        )

        return ex


    def _pad_length(self, row):
        # Pad mistake inputs
        # check if mistake_inputs exist
        if "mistake_inputs" in row:
            mistake_inputs = row["mistake_inputs"][: self.mel_length].to(torch.float32)
            if mistake_inputs.shape[0] < self.mel_length:
                pad = torch.zeros(
                    self.mel_length - mistake_inputs.shape[0],
                    mistake_inputs.shape[1],
                    dtype=mistake_inputs.dtype,
                    device=mistake_inputs.device,
                )
                mistake_inputs = torch.cat([mistake_inputs, pad], dim=0)
        else:
            mistake_inputs = None
            mistake_input_times = None

        # Pad score inputs
        if "score_inputs" in row:
            score_inputs = row["score_inputs"][: self.mel_length].to(torch.float32)
            if score_inputs.shape[0] < self.mel_length:
                pad = torch.zeros(
                    self.mel_length - score_inputs.shape[0],
                    score_inputs.shape[1],
                    dtype=score_inputs.dtype,
                    device=score_inputs.device,
                )
                score_inputs = torch.cat([score_inputs, pad], dim=0)
        else:
            score_inputs = None
            score_input_times = None

        # Pad targets
        if "targets" in row:
            targets = torch.from_numpy(row["targets"][: self.event_length]).to(torch.long)
            targets = targets + self.vocab.num_special_tokens()
            if targets.shape[0] < self.event_length:
                eos = torch.ones(1, dtype=targets.dtype, device=targets.device)
                if self.event_length - targets.shape[0] - 1 > 0:
                    pad = (
                        torch.ones(
                            self.event_length - targets.shape[0] - 1,
                            dtype=targets.dtype,
                            device=targets.device,
                        )
                        * -100
                    )
                    targets = torch.cat([targets, eos, pad], dim=0)
                else:
                    targets = torch.cat([targets, eos], dim=0)
        else:
            targets = None
            

        if "prompt_events" in row:
            if not isinstance(row["prompt_events"], np.ndarray):
                row["prompt_events"] = np.array(row["prompt_events"])  # Convert to NumPy array
            prompts = torch.from_numpy(row["prompt_events"][: self.prompt_length]).to(torch.long)
            prompts = prompts + self.vocab.num_special_tokens()
            if prompts.shape[0] < self.prompt_length:
                if self.prompt_length - prompts.shape[0] > 0:
                    pad = (
                        torch.ones(
                            self.prompt_length - prompts.shape[0],
                            dtype=prompts.dtype,
                            device=prompts.device,
                        )
                        * -100
                    )
                    prompts = torch.cat([prompts, pad], dim=0)
            prompts_attention_mask = (prompts != -100).long()
        else:
            print("No prompts found", flush=True)
            prompts = None
            prompts_attention_mask = None

        return {
            "mistake_inputs": mistake_inputs,
            "score_inputs": score_inputs,
            "targets": targets,
            "prompts": prompts,
            "prompts_attention_mask": prompts_attention_mask,
        }


    def _split_frame(self, row, length=2000):
        rows = []
        input_length = row["score_inputs"].shape[0]

        # Chunk the audio into chunks of `length`
        for split in range(0, input_length, length):
            if split + length >= input_length:
                continue
            new_row = {}
            # is splitting like this valid? the indices are not altered though
            for k in row.keys():
                if k in [
                    "mistake_inputs",
                    "mistake_input_times",
                    "score_inputs",
                    "score_input_times",
                    "input_event_start_indices",
                    "input_event_end_indices",
                    "input_state_event_indices",
                    "prompt_event_start_indices",
                    "prompt_event_end_indices",
                    "prompt_state_event_indices",
                ]:
                    new_row[k] = row[k][split : split + length]
                else:
                    new_row[k] = row[k]
            rows.append(new_row)

        if len(rows) == 0:
            return [row]
        return rows

    def _random_chunk(self, row):
        input_length = row["mistake_inputs"].shape[0]
        random_length = input_length - self.mel_length 
        # Calculate the total amount of context needed around the slice
        extra_context_window = self.context_multiplier - 1
        if random_length < 1:
            start_length = 0  # Start from the beginning if not enough length
            padding_length = self.mel_length - input_length
        else:
            start_length = (
                random.randint(0, random_length) if not self.is_deterministic else 0
            )
            padding_length = 0

        
        if self.random_alignment_shift_augmentation:
            random_shift = random.randint(
                -int(self.mel_length * 0.5 * extra_context_window), int(self.mel_length * 0.5 * extra_context_window)
            )
        else:
            random_shift = 0
        # Calculate start and end indices for extended context
        extended_start = max(
            0, start_length - int(self.mel_length * 0.5 * extra_context_window) + random_shift
        )
        extended_end = min(
            input_length,
            start_length
            + self.mel_length
            + int(self.mel_length * 0.5 * extra_context_window) + random_shift,
        )

        # Calculate padding if necessary
        start_padding = extended_start - (
            start_length - int(self.mel_length * 0.5 * extra_context_window)
        )
        end_padding = (
            start_length
            + self.mel_length
            + int(self.mel_length * 0.5 * extra_context_window)
        ) - extended_end

        new_row = {}
        # Doing this fixes a bug where the indices are padded with zeros
        score_keys = ["score_inputs", "score_input_times"]
        prompt_keys = [
            "prompt_event_start_indices",
            "prompt_event_end_indices",
            "prompt_state_event_indices",
        ]
        for k in row.keys():
            if k in [
                "mistake_inputs",
                "mistake_input_times",
                "input_event_start_indices",
                "input_event_end_indices",
                "input_state_event_indices",
            ]:
                # Standard context without extended padding
                slice_start = start_length
                slice_end = start_length + self.mel_length
                data_slice = row[k][slice_start:slice_end]
                if isinstance(data_slice, np.ndarray):
                    data_slice = torch.from_numpy(data_slice)
                if padding_length > 0:
                    # print("padding targets", padding_length, flush=True)
                    if len(data_slice.shape) == 2:
                        padding = (0, 0, 0, padding_length)
                    if len(data_slice.shape) == 1:
                        # right padding
                        padding = (0, padding_length)
                    data_slice = F.pad(data_slice, padding, "constant", 0).squeeze(0)
            elif k in score_keys:
                # Extended context for score inputs + padding
                data_slice = row[k][extended_start:extended_end]
                if isinstance(data_slice, np.ndarray):
                    data_slice = torch.from_numpy(data_slice)
                if start_padding > 0 or end_padding > 0:
                    if len(data_slice.shape) == 2:
                        padding = (0, 0, max(0, start_padding), max(0, end_padding))
                    else:
                        padding = (max(0, start_padding), max(0, end_padding))
                    data_slice = F.pad(data_slice, padding, "constant", 0).squeeze(0)
            elif k in prompt_keys:
                # Extended context for prompt event indices but NO padding
                data_slice = row[k][extended_start:extended_end]
                if isinstance(data_slice, np.ndarray):
                    data_slice = torch.from_numpy(data_slice)
                # Skip the “if start_padding > 0...” block here
            else:
                # For everything else
                data_slice = row[k]

            new_row[k] = data_slice
            

        return new_row


    def _postprocess_batch(self, result):
        EOS_TOKEN_ID = 1  # TODO: this is a hack!
        after_eos = torch.cumsum((result == EOS_TOKEN_ID).float(), dim=-1)
        # minus special token
        result = result - self.vocab.num_special_tokens()
        result = torch.where(after_eos.bool(), -1, result)
        # remove bos
        result = result[:, 1:]
        result = result.cpu().numpy()
        return result

    def _preprocess_inputs(self, row):
        """
        Load and parse MIDI and audio files for given dataset entry.

        Args:
            row (dict): Dictionary containing paths to MIDI and audio files.

        Returns:
            Tuple containing:
                - A list of note sequences parsed from the MIDI files.
                - The audio waveform corresponding to the 'mistake' audio file.
                - The audio waveform corresponding to the 'score' audio file.
        """
        # Example check for 'mistake_audio'
        mistake_audio_path = row.get("mistake_audio")
        if mistake_audio_path and os.path.exists(mistake_audio_path):
            mistake_audio, sr_mistake = librosa.load(mistake_audio_path, sr=None)
        else:
            print(f"File not found: {mistake_audio_path}")
            return [None, None, None], None, None

        # Repeat the check for 'score_audio'
        score_audio_path = row.get("score_audio")
        if score_audio_path and os.path.exists(score_audio_path):
            score_audio, sr_score = librosa.load(score_audio_path, sr=None)
        else:
            print(f"File not found: {score_audio_path}")
            return [None, None, None], None, None
        extra_notes_ns = note_seq.midi_file_to_note_sequence(row["extra_notes_midi"])
        removed_notes_ns = note_seq.midi_file_to_note_sequence(
            row["removed_notes_midi"]
        )
        mistake_ns = note_seq.midi_file_to_note_sequence(row["correct_notes_midi"])
        score_midi_path = self.lowercase_instrument_name(row["score_audio"].replace(".wav", ".mid").replace("stems_audio", "stems_midi"))
        score_ns = note_seq.midi_file_to_note_sequence(score_midi_path)
        

        # Load audio files
        mistake_audio, sr_mistake = librosa.load(row["mistake_audio"], sr=None)
        score_audio, sr_score = librosa.load(row["score_audio"], sr=None)

        # Resample audio if necessary
        if sr_mistake != self.spectrogram_config.sample_rate:
            mistake_audio = librosa.resample(
                mistake_audio,
                orig_sr=sr_mistake,
                target_sr=self.spectrogram_config.sample_rate,
            )
        if sr_score != self.spectrogram_config.sample_rate:
            score_audio = librosa.resample(
                score_audio,
                orig_sr=sr_score,
                target_sr=self.spectrogram_config.sample_rate,
            )

        # Return tuple of note sequences and audio data
        return (
            [extra_notes_ns, removed_notes_ns, mistake_ns, score_ns],
            mistake_audio,
            score_audio,
        )

    def __getitem__(self, idx):
        # Preprocess inputs to get the tracks and audio data
        tracks, mistake_audio, score_audio = self._preprocess_inputs(self.df[idx])

        # Tokenize the tracks and audio data
        row = self._tokenize(
            tracks,
            mistake_audio,
            score_audio,
            None,
        )

        # Split the data into frames/chunks
        rows = self._split_frame(row, length=self.split_frame_length)

        # Initialize lists to collect processed data
        mistake_inputs = []
        score_inputs = []
        targets = []
        prompts = []
        prompts_attention_mask = []


        # Limit the number of rows per batch by randomly getting a subset
        if len(rows) > self.num_rows_per_batch:
            if self.is_deterministic:
                start_idx = 0
            else:
                start_idx = random.randint(0, len(rows) - self.num_rows_per_batch)
            rows = rows[start_idx: start_idx + self.num_rows_per_batch]

        # Process each row
        for j, row in enumerate(rows):
            # Randomly select a chunk from the data
            row = self._random_chunk(row)
            # print("Row after random chunk:", rows, flush=True)

            # Extract the target sequence corresponding to the audio token segment
            row = self._extract_target_sequence_with_indices(row, self.tie_token)

            # print("Row after extract target indices:", rows, flush=True)

            # Run-length encode the shift events
            row = self.run_length_encode_shifts(row)
            row = self.run_length_encode_shifts(row, feature_key="prompt_events")
            # print("Row after length encode shifts:", rows, flush=True)

            # Debugging: print the targets after run-length encoding
            # print(f"\n--- Targets after run-length encoding for row {j} ---")
            # print([self.get_token_name(t) for t in row["targets"]])
            
            # print(f"\n--- Prompt events after run-length encoding for row {j} ---")
            # print([self.get_token_name(t) for t in row["prompt_events"]])

            # Compute the spectrograms for the inputs
            row = self._compute_spectrogram(row)

            # Apply random order augmentation if enabled
            if self.is_randomize_tokens:
                t = self.randomize_tokens(
                    [self.get_token_name(t) for t in row["targets"]]
                )
                t = np.array([self.token_to_idx(k) for k in t])
                t = self.remove_redundant_tokens(t)
                row["targets"] = t
                
            # if self.is_randomize_tokens:
            #     t = self.randomize_tokens(
            #         [self.get_token_name(t) for t in row["prompt_events"]]
            #     )
            #     t = np.array([self.token_to_idx(k) for k in t])
            #     t = self.remove_redundant_tokens(t)
            #     row["prompt_events"] = t

                # Debugging: print the targets after random order augmentation
                # print(f"\n--- Targets after random order augmentation for row {j} ---")
                # print([self.get_token_name(t) for t in row["targets"]])

            # Pad the inputs and targets to the required length
            row = self._pad_length(row)
            # print("after pad_length, prompt tokens (first 100):", row["prompts"][:100], flush=True)
            # print("after pad_length, prompt attention mask (first 100):", row["prompts_attention_mask"][:100], flush=True)
            mistake_inputs.append(row["mistake_inputs"])
            score_inputs.append(row["score_inputs"])
            targets.append(row["targets"])
            prompts.append(row["prompts"])
            prompts_attention_mask.append(row["prompts_attention_mask"])
            

        
        # Convert targets to torch tensors if not already
        targets = [t if isinstance(t, torch.Tensor) else torch.from_numpy(t).long() for t in targets]
        prompts = [p if isinstance(p, torch.Tensor) else torch.from_numpy(p).long() for p in prompts]


        if self.use_prompt:
            # Return data with prompts
            return torch.stack(mistake_inputs), torch.stack(score_inputs), torch.stack(targets), torch.stack(prompts), torch.stack(prompts_attention_mask)
        else:
            # Return data without prompts
            return torch.stack(mistake_inputs), torch.stack(score_inputs), torch.stack(targets)


    def __len__(self):
        return len(self.df)

    def randomize_tokens(self, token_lst):
        shift_idx = [i for i in range(len(token_lst)) if "shift" in token_lst[i]]
        if len(shift_idx) == 0:
            return token_lst
        res = token_lst[: shift_idx[0]]
        for j in range(len(shift_idx) - 1):
            res += [token_lst[shift_idx[j]]]

            start_idx = shift_idx[j]
            end_idx = shift_idx[j + 1]
            cur = token_lst[start_idx + 1 : end_idx]
            cur_lst = []
            ptr = 0
            # print("cur", cur)
            # print(f"len(cur) {len(cur)}")
            while ptr < len(cur):
                t = cur[ptr]
                if "error" in t:
                    # print("error_class", t)
                    cur_lst.append([cur[ptr], cur[ptr + 1], cur[ptr + 2]])
                    ptr += 3
                elif "velocity" in t:
                    # print("velocity", t)
                    cur_lst.append([cur[ptr], cur[ptr + 1]])
                    ptr += 2
                else:
                    raise Exception("infinte loop detected, check get_token_name names")
                # print("ptr", ptr)

            indices = np.arange(len(cur_lst))
            np.random.shuffle(indices)

            new_cur_lst = []
            for idx in indices:
                new_cur_lst.append(cur_lst[idx])

            new_cur_lst = [item for sublist in new_cur_lst for item in sublist]
            res += new_cur_lst

        res += token_lst[shift_idx[-1] :]
        return res

    def get_token_name(self, token_idx):
        token_idx = int(token_idx)
        if token_idx >= 1001 and token_idx <= 1128:
            return f"pitch_{token_idx - 1001}"
        elif token_idx >= 1129 and token_idx <= 1130:
            return f"velocity_{token_idx - 1129}"
        elif token_idx == 1131:
            return "tie"
        elif token_idx >= 1132 and token_idx <= 1134:
            return f"error_{token_idx - 1132}"  # Adjusted to the new range for 3 error types
        elif token_idx >= 0 and token_idx < 1000:
            return f"shift_{token_idx}"
        else:
            return f"invalid_{token_idx}"

        return token

    def token_to_idx(self, token_str):
        if "pitch" in token_str:
            token_idx = int(token_str.split("_")[1]) + 1001
        elif "velocity" in token_str:
            token_idx = int(token_str.split("_")[1]) + 1129
        elif "tie" in token_str:
            token_idx = 1131
        elif "error" in token_str:
            token_idx = int(token_str.split("_")[1]) + 1132
        elif "shift" in token_str:
            token_idx = int(token_str.split("_")[1])
        else:
            raise ValueError("Unknown token string: {}".format(token_str))

        return token_idx


def collate_fn(lst):
    """
    Custom collate function to handle batches where some items may have either
    three or five elements.

    Args:
        lst (list): List of tuples, each containing either 3 or 5 tensors.

    Returns:
        tuple or None:
            - If all items have 5 elements, returns a tuple of five concatenated tensors:
                (mistake_inputs, score_inputs, targets, decoder_input_ids, decoder_attention_mask)
            - If all items have only 3 elements, returns a tuple of three concatenated tensors:
                (mistake_inputs, score_inputs, targets)
            - If items have mixed lengths or unexpected sizes, returns None.
    """
    # Filter out None entries which may be returned if files are missing or any error occurs
    filtered_lst = [item for item in lst if item is not None]

    if not filtered_lst:
        # All items are None; return None to indicate an empty batch
        return None

    # Determine the unique lengths of items in the batch
    unique_lengths = set(len(item) for item in filtered_lst)

    if len(unique_lengths) != 1:
        # Mixed lengths detected; cannot process reliably
        print("Warning: Mixed item lengths detected in batch. Returning None.")
        return None

    item_length = unique_lengths.pop()

    if item_length == 5:
        try:
            # Unpack and concatenate all five elements
            mistake_inputs = torch.cat([item[0] for item in filtered_lst], dim=0)
            score_inputs = torch.cat([item[1] for item in filtered_lst], dim=0)
            targets = torch.cat([item[2] for item in filtered_lst], dim=0)
            decoder_input_ids = torch.cat([item[3] for item in filtered_lst], dim=0)
            decoder_attention_mask = torch.cat([item[4] for item in filtered_lst], dim=0)

            return (
                mistake_inputs,
                score_inputs,
                targets,
                decoder_input_ids,
                decoder_attention_mask,
            )
        except IndexError as e:
            # Handle unexpected indexing issues
            print(f"Error during concatenation: {e}. Returning None.")
            return None

    elif item_length == 3:
        try:
            # Unpack and concatenate the three elements
            mistake_inputs = torch.cat([item[0] for item in filtered_lst], dim=0)
            score_inputs = torch.cat([item[1] for item in filtered_lst], dim=0)
            targets = torch.cat([item[2] for item in filtered_lst], dim=0)

            return (
                mistake_inputs,
                score_inputs,
                targets,
            )
        except IndexError as e:
            # Handle unexpected indexing issues
            print(f"Error during concatenation: {e}. Returning None.")
            return None

    else:
        # Unexpected number of elements; return None
        print(f"Warning: Unexpected item length {item_length}. Returning None.")
        return None


def plot_spectrogram(spectrogram, idx, title, file_suffix):
    plt.figure(figsize=(10, 4))
    # Transpose the spectrogram to have time on the x-axis and frequency on the y-axis
    spectrogram = spectrogram.T  # Transpose to flip axes
    plt.imshow(spectrogram, aspect="auto", origin="lower", cmap="viridis")
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"{title} for Sample {idx}")
    plt.xlabel("Time Frame")
    plt.ylabel("Mel Frequency")
    plt.savefig(f"{file_suffix}_{idx}.png")
    plt.close()
    print(f"Saved {title.lower()} image for sample {idx}.")


if __name__ == "__main__":
    root_dir = os.environ.get("LADDERSYM_COCO_ROOT")
    split_json_path = os.environ.get("LADDERSYM_COCO_SPLIT_JSON")
    if not root_dir or not split_json_path:
        raise SystemExit(
            "Set LADDERSYM_COCO_ROOT and LADDERSYM_COCO_SPLIT_JSON "
            "before running this dataset debug entrypoint."
        )

    # Parameters
    params = {
        "root_dir": root_dir,
        "split_json_path": split_json_path,
        "split": "train",
        "mel_length": 256,
        "event_length": 1024,
        "midi_folder": "MIDI",
        "audio_filename": "mix_16k.wav",
        "num_rows_per_batch": 12,
        "split_frame_length": 2000,
        "is_deterministic": False,
        "is_randomize_tokens": True,
        "use_prompt": True,
    }

    # Initialize Dataset
    dataset = Dataset(
        root_dir=params["root_dir"],
        split_json_path=params["split_json_path"],
        split=params["split"],
        mel_length=params["mel_length"],
        event_length=params["event_length"],
        midi_folder=params["midi_folder"],
        audio_filename=params["audio_filename"],
        num_rows_per_batch=params["num_rows_per_batch"],
        split_frame_length=params["split_frame_length"],
        is_deterministic=params["is_deterministic"],
        is_randomize_tokens=params["is_randomize_tokens"],
        use_prompt=params["use_prompt"],
    )

    print("Dataset initialized.")
    print("pitch range:", dataset.codec.event_type_range("pitch"))
    print("velocity range:", dataset.codec.event_type_range("velocity"))
    print("tie range:", dataset.codec.event_type_range("tie"))
    print("error_class range:", dataset.codec.event_type_range("error_class"))

    # DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=4,
        collate_fn=collate_fn,
        shuffle=False,
    )

    # Iterate through DataLoader
    for idx, item in enumerate(dataloader):
        if item is None:
            print(f"Batch {idx}: Data is None. Skipping.")
            continue

        if len(item) == 5:
            mistake_inputs, score_inputs, targets, decoder_input_ids, decoder_attention_mask = item
        elif len(item) == 3:
            mistake_inputs, score_inputs, targets = item
            decoder_input_ids = decoder_attention_mask = None
        else:
            print(f"Batch {idx}: Unexpected item length. Skipping.")
            continue

        # Debugging shapes
        print(f"Batch {idx} - Shapes:")
        print(f"  mistake_inputs: {mistake_inputs.shape}")
        print(f"  score_inputs: {score_inputs.shape}")
        print(f"  targets: {targets.shape}")

        if decoder_input_ids is not None:
            print(f"  decoder_input_ids: {decoder_input_ids.shape}")
            print(f"  decoder_attention_mask: {decoder_attention_mask.shape}")

        # Plot spectrograms
        mistake_spectrogram = mistake_inputs.squeeze().numpy()
        score_spectrogram = score_inputs.squeeze().numpy()

        if mistake_spectrogram.ndim == 2:
            plot_spectrogram(
                mistake_spectrogram,
                idx,
                "Mel Spectrogram (Mistake Inputs)",
                "mel_spectrogram_mistake",
            )

        if score_spectrogram.ndim == 2:
            plot_spectrogram(
                score_spectrogram,
                idx,
                "Mel Spectrogram (Score Inputs)",
                "mel_spectrogram_score",
            )

        # Decode targets
        targets_numpy = targets.numpy()
        decoded_tokens = [dataset.get_token_name(token - 3) for token in targets_numpy[0]]

        print(f"Batch {idx} - Decoded Tokens:")
        print(decoded_tokens[:100])  # Print first 100 tokens for readability

        # Decode prompt if available
        if decoder_input_ids is not None:
            decoder_numpy = decoder_input_ids.squeeze().numpy()
            decoded_prompt = [dataset.get_token_name(token - 3) for token in decoder_numpy]

            print(f"Batch {idx} - Decoded Prompt:")
            print(decoded_prompt[:100])  # Print first 100 prompt tokens for readability

        if idx == 0:  # Limit to the first batch for debugging
            break
