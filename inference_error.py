import json
import math
import os
from typing import List
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch
from contrib import spectrograms as spectrograms
from contrib import vocabularies, note_sequences, metrics_utils, run_length_encoding, note_sequences
from contrib.preprocessor import (
    class_to_error,
    add_track_to_notesequence,
    PitchBendError,
)
from dataset.dataset_2_random import Dataset
import note_seq
import traceback
import matplotlib.pyplot as plt

MIN_LOG_MEL = -12
MAX_LOG_MEL = 5

dataset = Dataset(skip_build=True)


class InferenceHandler:

    def __init__(
        self,
        model=None,
        weight_path=None,
        device=torch.device("cuda"),
        mel_norm=True,
        contiguous_inference=False,
    ) -> None:
        self.model = model
        self.device = device
        if model is not None:
            self.model.to(self.device)

        self.contiguous_inference = contiguous_inference
        self.SAMPLE_RATE = 16000
        self.spectrogram_config = spectrograms.SpectrogramConfig()
        self.codec = vocabularies.build_codec(
            vocab_config=vocabularies.VocabularyConfig(num_velocity_bins=1)
        )
        self.vocab = vocabularies.vocabulary_from_codec(self.codec)
        self.mel_norm = mel_norm

        self.include_ties = dataset.include_ties
        # Keep a reference to the original prompt NoteSequence so we can
        # transfer its global timing metadata (tempo, time signature, TPQN)
        # onto the decoded output before writing to MIDI.
        self._prompt_reference_ns = None
        # Likewise track the original mistake NoteSequence so we can align the
        # generated output with the mistake MIDI tempo map when desired.
        self._mistake_reference_ns = None

    def get_prompt(self, prompt_path, frame_times_score=None):
        """
        Extracts the prompt tokens and attention mask using logic from the Dataset.
        """
        
        """
        Extracts prompt tokens and attention mask.
        Adjusts MIDI duration by scaling event times.
        """
        # , tempo_factor=1.5)
        score_note_sequence = note_seq.midi_file_to_note_sequence(prompt_path)
        # Stash the original NoteSequence for later so we can restore its
        # tempo map (and other global state) on the decoded output.
        self._prompt_reference_ns = score_note_sequence
        ns = note_seq.NoteSequence(ticks_per_quarter=220)
        add_track_to_notesequence(ns, score_note_sequence, error_class=3, ignore_pitch_bends=True)
        note_sequences.assign_error_classes(ns)
        note_sequences.validate_note_sequence(ns)

        times, values = note_sequences.note_sequence_to_onsets_and_offsets_and_programs(ns)

        # # Scale the event times.
        # times = np.array(times) * tempo_factor
        duration = max(times) if len(times) > 0 else 0.0
        print("MIDI length in seconds:", duration, flush=True)
        if len(times) > 0:
            print(
                "Prompt event stats: count={}, first={}s, last={}s".format(
                    len(times), round(times[0], 6), round(times[-1], 6)
                ),
                flush=True,
            )
        if frame_times_score is not None and len(frame_times_score) > 0:
            hop = (
                float(frame_times_score[1] - frame_times_score[0])
                if len(frame_times_score) > 1
                else 0.0
            )
            print(
                "Score frame grid: count={}, first5={}, last5={}, hop={}s".format(
                    len(frame_times_score),
                    np.round(frame_times_score[:5], 6),
                    np.round(frame_times_score[-5:], 6),
                    round(hop, 6),
                ),
                flush=True,
            )
            if len(times) > 0:
                print(
                    "Frame grid coverage minus prompt duration: {}s".format(
                        round(
                            (frame_times_score[-1] + hop) - duration,
                            6,
                        )
                    ),
                    flush=True,
                )
        
        if frame_times_score is None:
            print("frame_times_score is None", flush=True)
            frame_times_score = np.zeros(len(times))
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
        # print("prompt_events shape", prompt_events.shape, "prompt_event_start_indices", prompt_event_start_indices.shape, "prompt_event_end_indices", prompt_event_end_indices.shape, "prompt_state_events", prompt_state_events.shape, "prompt_state_event_indices", prompt_state_event_indices.shape, flush=True)
        sos_token = self.model.config.decoder_start_token_id
        prompt_events = np.insert(prompt_events, len(prompt_events), sos_token)

        prompt_row = {
            "targets": prompt_events,
            "input_event_start_indices": prompt_event_start_indices,
            "input_event_end_indices": prompt_event_end_indices,
            "state_events": prompt_state_events,
            "input_state_event_indices": prompt_state_event_indices,
        }
        
        return prompt_row

    
    def _extract_target_sequence_with_indices(self, batch_features, state_events_end_token=None):
        """
        Extract target and state event sequences for a batched input based on batched indices.

        Args:
            batch_features (dict): A dictionary containing batched feature data.
            state_events_end_token (optional): Token indicating the end of state events.

        Returns:
            Updated batch_features with extracted target and state event sequences for each batch.
        """
        # Initialize containers for batched results
        updated_targets = []
        updated_state_events = []

        # Iterate through each batch entry
        for i in range(len(batch_features["prompt_event_start_indices"])):
            # Extract start and end indices for the current batch entry
            prompt_start_idx = int(batch_features["prompt_event_start_indices"][i][0])
            prompt_end_idx = int(batch_features["prompt_event_end_indices"][i][-1])

            # Extract the target segment
            targets = batch_features["targets"][prompt_start_idx:prompt_end_idx]

            # Extract and prepend state events if specified
            if state_events_end_token is not None:
                prompt_state_event_start_idx = int(batch_features["prompt_state_event_indices"][i][0])
                prompt_state_event_end_idx = prompt_state_event_start_idx + 1

                # Extend until the end token or the sequence end
                while (
                    batch_features["prompt_state_events"][prompt_state_event_end_idx - 1]
                    != state_events_end_token
                ):
                    prompt_state_event_end_idx += 1
                    if prompt_state_event_end_idx >= len(batch_features["prompt_state_events"]):
                        print(f"Encountered end of state events without end token for batch index {i}.", flush=True)
                        break

                # Prepend state events to the targets
                state_events = batch_features["prompt_state_events"][
                    prompt_state_event_start_idx:prompt_state_event_end_idx
                ]
                targets = np.concatenate([state_events, targets], axis=0)

            # Append the processed targets and state events to the batch results
            updated_targets.append(targets)
            updated_state_events.append(batch_features["prompt_state_events"])

        # Update the batch_features dictionary with the new results
        batch_features["targets"] = updated_targets
        batch_features["prompt_state_events"] = updated_state_events

        return batch_features
    
    def _audio_to_frames(self, audio):
        """Compute spectrogram frames from audio."""
        spectrogram_config = self.spectrogram_config
        frame_size = spectrogram_config.hop_width
        padding = [0, frame_size - len(audio) % frame_size]
        audio = np.pad(audio, padding, mode='constant')
        frames = spectrograms.split_audio(audio, spectrogram_config)
        num_frames = len(audio) // frame_size
        times = np.arange(num_frames) / spectrogram_config.frames_per_second
        return frames, times
    
    def _split_token_into_length(
        self,
        mistake_frames,
        score_frames,
        mistake_frame_times,
        score_frame_times,
        features,
        max_length=256,
        return_prompt_row=False
    ):
        """
        Batch 1: [Frame0, Frame1, Frame2, Frame3] (no padding needed)
        Batch 2: [Frame4, Frame5, Frame6, Frame7] (no padding needed)
        Batch 3: [Frame8, Frame9, Pad, Pad]      (2 mistake_frames padded)
        max_length: maximum number of frames in a batch
        """
        assert len(mistake_frames.shape) >= 1
        assert mistake_frames.shape[0] == mistake_frame_times.shape[0], (
            "Mismatch between mistake_frames and mistake_frame_times lengths"
        )
        # print("mistake_frames", mistake_frames.shape, "score_frames", score_frames.shape, "mistake_frame_times", mistake_frame_times.shape)
        assert len(score_frames.shape) >= 1
        assert score_frames.shape[0] == score_frame_times.shape[0], (
            "Mismatch between mistake_frames and score_frame_times lengths"
        )
        # print("mistake_frames", mistake_frames.shape, "score_frames", score_frames.shape, "score_frame_times", score_frame_times.shape)

        # Find the max frame shape
        max_frame_shape = max(mistake_frames.shape[0], score_frames.shape[0])
        
        # Pad the frames to be of equal length
        if mistake_frames.shape[0] < max_frame_shape:
            mistake_frames = np.pad(mistake_frames, ((0, max_frame_shape - mistake_frames.shape[0]), (0, 0)), mode='constant')
        if score_frames.shape[0] < max_frame_shape:
            score_frames = np.pad(score_frames, ((0, max_frame_shape - score_frames.shape[0]), (0, 0)), mode='constant')
        # print("last 100 mistake_frames", mistake_frames[-100:], "last 100 score_frames", score_frames[-100:], flush=True)
        # print("last 100 mistake_frame_times", mistake_frame_times[-100:], "last 100 score_frame_times", score_frame_times[-100:], flush=True)
        
        num_segment = math.ceil(mistake_frames.shape[0] / max_length)  # Use mistake_frames shape here
        mistake_batches = []
        score_batches = []
        mistake_frame_times_batches = []
        score_frame_times_batches = []
        mistake_paddings = []
        score_paddings = []
        if return_prompt_row is True:
            prompt_events_start_indices_batches = []
            prompt_events_end_indices_batches = []
            prompt_state_events_indices_batches = []
            
            # Retrieve event indices from features:
            prompt_event_start_indices = features["input_event_start_indices"]
            prompt_event_end_indices = features["input_event_end_indices"]
            prompt_state_event_indices = features["input_state_event_indices"]

            # Pad prompt-related features (to match score_frames)
            if len(features["input_event_start_indices"]) < max_frame_shape:
                prompt_event_start_indices = np.pad(
                    features["input_event_start_indices"],
                    (0, max_frame_shape - len(features["input_event_start_indices"])),
                    mode='constant'
                )
            else:
                prompt_event_start_indices = features["input_event_start_indices"]

            if len(features["input_event_end_indices"]) < max_frame_shape:
                prompt_event_end_indices = np.pad(
                    features["input_event_end_indices"],
                    (0, max_frame_shape - len(features["input_event_end_indices"])),
                    mode='constant'
                )
            else:
                prompt_event_end_indices = features["input_event_end_indices"]

            if len(features["input_state_event_indices"]) < max_frame_shape:
                prompt_state_event_indices = np.pad(
                    features["input_state_event_indices"],
                    (0, max_frame_shape - len(features["input_state_event_indices"])),
                    mode='constant'
                )
            else:
                prompt_state_event_indices = features["input_state_event_indices"]
                
            # print("mistake_frame_times", mistake_frame_times.shape, "score_frame_times", score_frame_times.shape, flush=True)
            # print("max_frame_shape", max_frame_shape, flush=True)
            
            # Pad frame times to match the padded frame length.
            # We extend the shorter array using its last available value.
            # TODO: add this back if testing shows it's fine
            # if mistake_frame_times.shape[0] < max_frame_shape:
            #     diff = max_frame_shape - mistake_frame_times.shape[0]
            #     mistake_frame_times = np.pad(mistake_frame_times, (0, diff), mode='edge')
            # if score_frame_times.shape[0] < max_frame_shape:
            #     diff = max_frame_shape - score_frame_times.shape[0]
            #     score_frame_times = np.pad(score_frame_times, (0, diff), mode='edge')

            # print("shape after padding", mistake_frame_times.shape, score_frame_times.shape, flush=True)
            # print("prompt_event_start_indices", prompt_event_start_indices.shape, "prompt_event_end_indices", prompt_event_end_indices.shape, "prompt_state_event_indices", prompt_state_event_indices.shape, flush=True)
            # print("score_frame_times", score_frame_times.shape, flush=True)
            # print("score_frames", score_frames.shape, flush=True)
            assert len(prompt_event_start_indices) == len(score_frames), "Feature indices length mismatch"
            assert len(prompt_event_end_indices) == len(score_frames), "Feature indices length mismatch"
            assert len(prompt_state_event_indices) == len(score_frames), "Feature indices length mismatch"
        for i in range(num_segment):
            mistake_batch = np.zeros((max_length, *mistake_frames.shape[1:]))
            mistake_frame_times_batch = np.zeros((max_length))
            score_frame_times_batch = np.zeros((2 * max_length))
            score_batch = np.zeros((2 * max_length, *score_frames.shape[1:]))
            
            start_idx = i * max_length
            end_idx = min(max_length, mistake_frames.shape[0] - start_idx)  # Calculate end_idx based on mistake_frames
            
            score_start_idx = start_idx - max_length // 2
            score_end_idx = min(max_length * 2, score_frames.shape[0] - score_start_idx)
            
            start_padding = 0
            if score_start_idx < 0:
                start_padding = -score_start_idx
                score_start_idx = 0

            # Adjust end indices to avoid overflows after padding adjustments
            max_mistake_available = max(0, mistake_frames.shape[0] - start_idx)
            end_idx = max(0, min(end_idx, max_mistake_available))

            max_score_available = max(0, score_frames.shape[0] - score_start_idx)
            score_slice_len = max(0, min(score_end_idx, max_score_available))
            score_slice_len = min(score_slice_len, 2 * max_length - start_padding)

            # Check frame_times bounds
            if start_idx + end_idx > mistake_frame_times.shape[0]:
                end_idx = max(0, mistake_frame_times.shape[0] - start_idx)

            if score_start_idx + score_slice_len > score_frame_times.shape[0]:
                score_slice_len = max(0, score_frame_times.shape[0] - score_start_idx)

            if end_idx == 0 and score_slice_len == 0:
                continue

            mistake_batch[0:end_idx, ...] = mistake_frames[start_idx:start_idx + end_idx, ...]
            score_batch[start_padding:start_padding + score_slice_len, ...] = score_frames[score_start_idx:score_start_idx + score_slice_len, ...]
            
            # Adjust mistake_frame_times_batch to match mistake_frames segmentation
            mistake_frame_times_batch[0:end_idx] = mistake_frame_times[start_idx:start_idx + end_idx]
            score_frame_times_batch[start_padding:start_padding + score_slice_len] = score_frame_times[score_start_idx:score_start_idx + score_slice_len]
            
            mistake_batches.append(mistake_batch)
            score_batches.append(score_batch)
            mistake_frame_times_batches.append(mistake_frame_times_batch)
            score_frame_times_batches.append(score_frame_times_batch)
            mistake_paddings.append(end_idx)
            score_paddings.append(score_slice_len)
            if return_prompt_row is True:
                prompt_event_start_indices_batch = np.zeros((2 * max_length))
                prompt_event_end_indices_batch = np.zeros((2 * max_length))
                prompt_state_event_indices_batch = np.zeros((2 * max_length))

                prompt_event_start_indices_batch[start_padding:start_padding + score_slice_len] = prompt_event_start_indices[score_start_idx:score_start_idx + score_slice_len]
                prompt_event_end_indices_batch[start_padding:start_padding + score_slice_len] = prompt_event_end_indices[score_start_idx:score_start_idx + score_slice_len]
                prompt_state_event_indices_batch[start_padding:start_padding + score_slice_len] = prompt_state_event_indices[score_start_idx:score_start_idx + score_slice_len]

                prompt_events_start_indices_batches.append(prompt_event_start_indices_batch)
                prompt_events_end_indices_batches.append(prompt_event_end_indices_batch)
                prompt_state_events_indices_batches.append(prompt_state_event_indices_batch)
                
        if return_prompt_row is True:
        # print("frame_times")
            batch_features = {
                "targets": features["targets"],
                "prompt_event_start_indices": prompt_events_start_indices_batches,
                "prompt_event_end_indices": prompt_events_end_indices_batches,
                "prompt_state_events": features["state_events"],
                "prompt_state_event_indices": prompt_state_events_indices_batches,
            }
            # print shapes of stacked batches
            # print("mistake_batches", np.stack(mistake_batches, axis=0).shape, "score_batches", np.stack(score_batches, axis=0).shape, "mistake_frame_times_batches", np.stack(mistake_frame_times_batches, axis=0).shape, "score_frame_times_batches", np.stack(score_frame_times_batches, axis=0).shape, flush=True)
            # # print shape of stacked prompt batches
            # print("prompt_event_start_indices_batches", np.stack(prompt_events_start_indices_batches, axis=0).shape, "prompt_event_end_indices_batches", np.stack(prompt_events_end_indices_batches, axis=0).shape, "prompt_state_event_indices_batches", np.stack(prompt_state_events_indices_batches, axis=0).shape, flush=True)
        if return_prompt_row:
            return (
                np.stack(mistake_batches, axis=0),
                np.stack(score_batches, axis=0),
                np.stack(mistake_frame_times_batches, axis=0),
                np.stack(score_frame_times_batches, axis=0),
                mistake_paddings,
                score_paddings,
                batch_features
            )
        else:
            return (
                np.stack(mistake_batches, axis=0),
                np.stack(score_batches, axis=0),
                np.stack(mistake_frame_times_batches, axis=0),
                np.stack(score_frame_times_batches, axis=0),
                mistake_paddings,
                score_paddings
            )

    
    
    def _compute_spectrograms(self, mistake_inputs, score_inputs):
        mistake_outputs = []
 
        for i in mistake_inputs:

            samples = spectrograms.flatten_frames(
                i,
            )
            i = spectrograms.compute_spectrogram(samples, self.spectrogram_config)
            mistake_outputs.append(i)

        mistake_melspec= np.stack(mistake_outputs, axis=0)

        # add normalization
        # NOTE: for MT3 pretrained weights, we don't do mel_norm
        if self.mel_norm:
            mistake_melspec = np.clip(mistake_melspec, MIN_LOG_MEL, MAX_LOG_MEL)
            mistake_melspec = (mistake_melspec - MIN_LOG_MEL) / (MAX_LOG_MEL - MIN_LOG_MEL)
            
        score_outputs = []

        for i in score_inputs:

            samples = spectrograms.flatten_frames(
                i,
            )
            i = spectrograms.compute_spectrogram(samples, self.spectrogram_config, context_multiplier=2)
            score_outputs.append(i)
        
        score_melspec = np.stack(score_outputs, axis=0)
        
        if self.mel_norm:
            score_melspec = np.clip(score_melspec, MIN_LOG_MEL, MAX_LOG_MEL)
            score_melspec = (score_melspec - MIN_LOG_MEL) / (MAX_LOG_MEL - MIN_LOG_MEL)
        
        return mistake_melspec, score_melspec

    


    def _preprocess(self, mistake_audio, score_audio, prompt_path, mistake_midi_path=None):
        mistake_frames, mistake_frame_times = self._audio_to_frames(mistake_audio)
        # print("mistake_frames", mistake_frames.shape, "frame_times", frame_times.shape)
        score_frames, score_frame_times = self._audio_to_frames(score_audio)
        mistake_audio_duration = len(mistake_audio) / self.SAMPLE_RATE
        score_audio_duration = len(score_audio) / self.SAMPLE_RATE
        if len(mistake_frame_times) > 0:
            mistake_hop = (
                float(mistake_frame_times[1] - mistake_frame_times[0])
                if len(mistake_frame_times) > 1
                else 0.0
            )
            print(
                "Mistake audio: {:.3f}s, frames={}, last_frame_start={:.3f}s, hop={:.6f}s".format(
                    mistake_audio_duration,
                    mistake_frames.shape[0],
                    mistake_frame_times[-1],
                    mistake_hop,
                ),
                flush=True,
            )
        if len(score_frame_times) > 0:
            score_hop = (
                float(score_frame_times[1] - score_frame_times[0])
                if len(score_frame_times) > 1
                else 0.0
            )
            print(
                "Score audio: {:.3f}s, frames={}, last_frame_start={:.3f}s, hop={:.6f}s".format(
                    score_audio_duration,
                    score_frames.shape[0],
                    score_frame_times[-1],
                    score_hop,
                ),
                flush=True,
            )

        # Load the reference mistake NoteSequence if available so we can copy
        # its tempo map onto the decoded output later.
        if mistake_midi_path and os.path.exists(mistake_midi_path):
            try:
                self._mistake_reference_ns = note_seq.midi_file_to_note_sequence(
                    mistake_midi_path
                )
            except Exception as exc:
                print(
                    f"Failed to load mistake MIDI tempo map from {mistake_midi_path}: {exc}",
                    flush=True,
                )
                self._mistake_reference_ns = None
        else:
            self._mistake_reference_ns = None
        print("use prompt", self.model.config.use_prompt, flush=True)
        if self.model.config.use_prompt:
            
            prompt_row = self.get_prompt(prompt_path, frame_times_score=score_frame_times)
            (
                mistake_frames,
                score_frames,
                mistake_frame_times,
                score_frame_times,
                mistake_paddings,
                score_paddings,
                prompt_row
            ) = self._split_token_into_length(
                mistake_frames,
                score_frames,
                mistake_frame_times,
                score_frame_times,
                prompt_row,
                return_prompt_row=True
            )
            # print("prompt after split", prompt_row["targets"], flush=True)
            # # need to modify to handle split prompts

            # print("dataset.tie_token", dataset.tie_token, flush=True)

            prompt_row = self._extract_target_sequence_with_indices(prompt_row, dataset.tie_token)
            ##
            # print("prompt after extract", prompt_row["targets"], flush=True)
        else:
            # Just ignore prompts if not in use
            (
                mistake_frames,
                score_frames,
                mistake_frame_times,
                score_frame_times,
                mistake_paddings,
                score_paddings
            ) = self._split_token_into_length(
                mistake_frames,
                score_frames,
                mistake_frame_times,
                score_frame_times,
                {"targets": None, "state_events": None},
                return_prompt_row=False
            )
            prompt_row = {}
            # Make sure we do not accidentally reuse a previous prompt's
            # tempo metadata when prompts are disabled for this run.
            self._prompt_reference_ns = None


        mistake_inputs, score_inputs = self._compute_spectrograms(mistake_frames, score_frames)
        # print("mistake_inputs", mistake_inputs.shape, "score_inputs", score_inputs.shape)
        for i, p in enumerate(mistake_paddings):

            mistake_inputs[i, p+1:] = 0 
            
        for i, p in enumerate(score_paddings):

            score_inputs[i, int(p/2)+2:] = 0
            
        return mistake_inputs, score_inputs, mistake_frame_times, score_frame_times, prompt_row

    def _batching(
        self,
        mistake_tensors,
        score_tensors,
        mistake_frame_times,
        score_frame_times,
        tokenized_prompt=None,
        prompt_attention_mask=None,
        batch_size=1,
    ):
        """
        Batch inputs, reusing prompt tokens and attention masks across batches,
        split consistently using a manual slicing approach.
        """
        # Initialize batches
        mistake_batches = []
        score_batches = []
        mistake_frame_times_batch = []
        score_frame_times_batch = []
        prompt_tokens_batch = []
        prompt_attention_mask_batch = []

        # Iterate in batches using start_idx and end_idx
        for start_idx in range(0, mistake_tensors.shape[0], batch_size):
            end_idx = min(start_idx + batch_size, mistake_tensors.shape[0])

            # Slice tensors and append to respective batches
            mistake_batches.append(mistake_tensors[start_idx:end_idx].to(self.device))
            score_batches.append(score_tensors[start_idx:end_idx].to(self.device))
            mistake_frame_times_batch.append(mistake_frame_times[start_idx:end_idx])
            score_frame_times_batch.append(score_frame_times[start_idx:end_idx])

            # Handle optional prompts, slicing only if they are provided
            if tokenized_prompt is not None and prompt_attention_mask is not None:
                prompt_tokens_batch.append(tokenized_prompt[start_idx:end_idx].to(self.device))
                prompt_attention_mask_batch.append(prompt_attention_mask[start_idx:end_idx].to(self.device))

        # If prompts are not provided, fill batches with None
        if not prompt_tokens_batch:
            prompt_tokens_batch = [None] * len(mistake_batches)
        if not prompt_attention_mask_batch:
            prompt_attention_mask_batch = [None] * len(mistake_batches)

        return (
            mistake_batches,
            score_batches,
            mistake_frame_times_batch,
            score_frame_times_batch,
            prompt_tokens_batch,
            prompt_attention_mask_batch,
        )
    
    def _get_program_ids(self, valid_programs) -> List[List[int]]:
        min_program_id, max_program_id = self.codec.event_type_range("program")
        total_programs = max_program_id - min_program_id
        invalid_programs = []
        for p in range(total_programs):
            if p not in valid_programs:
                invalid_programs.append(p)
        invalid_programs = [min_program_id + id for id in invalid_programs]
        invalid_programs = self.vocab.encode(invalid_programs)
        return [[p] for p in invalid_programs]
    
    def _postprocess_prompt_batch(self, prompt_row, dataset):
        """
        For each subarray in prompt_row["targets"]:
          1) run_length_encode_shifts
          2) Print debug info (token names)
          3) Add special tokens
          4) Pad
          5) Convert to [1, seq_len] Tensors => prompt_tokens, prompt_masks

        Stores final arrays in:
          prompt_row["targets"]       (RLE + special offset)
          prompt_row["prompt_tokens"] (list of Tensors)
          prompt_row["prompt_masks"]  (list of Tensors)
        """

        new_targets = []
        all_prompt_tokens = []
        all_prompt_masks = []

        for idx, arr in enumerate(prompt_row["targets"]):
            # 1) Run-length encode the single subarray
            temp_features = {"prompt_events": arr}
            temp_features = dataset.run_length_encode_shifts(temp_features, feature_key="prompt_events")
            encoded_arr = temp_features["prompt_events"]

            # 2) Print debug info
            # print(f"Targets for item {idx} after RLE:", flush=True)
            # print([dataset.get_token_name(t) for t in encoded_arr], flush=True)
            
            # 3) remove redundant tokens (is this needed?)
            if dataset.is_randomize_tokens:
                encoded_arr = dataset.randomize_tokens(
                    [dataset.get_token_name(t) for t in encoded_arr]
                )
                encoded_arr = np.array([dataset.token_to_idx(k) for k in encoded_arr])
                encoded_arr = dataset.remove_redundant_tokens(encoded_arr)

            # 4) Pad
            #    We'll call dataset._pad_length, passing "prompt_events" = encoded_arr
            temp_row = {"prompt_events": encoded_arr}
            padded = dataset._pad_length(temp_row)
            
            # print("Prompt tokens after padding (first 100):", padded["prompts"][:100], flush=True)
            # print("Prompt attention mask after padding (first 100):", padded["prompts_attention_mask"][:100], flush=True)
            # 5) Convert to Tensors
            prompt_tokens = torch.tensor(padded["prompts"]).unsqueeze(0)
            prompt_tokens = prompt_tokens.masked_fill(
                prompt_tokens == -100,
                self.model.config.pad_token_id
            )
            prompt_mask = torch.tensor(padded["prompts_attention_mask"]).unsqueeze(0)

            # Collect final results
            new_targets.append(encoded_arr)
            all_prompt_tokens.append(prompt_tokens)
            all_prompt_masks.append(prompt_mask)

        # Overwrite prompt_row["targets"] with the final encoded arrays
        prompt_row["targets"] = new_targets
        # Store tokens & masks
        prompt_row["prompt_tokens"] = all_prompt_tokens
        prompt_row["prompt_masks"] = all_prompt_masks

        return prompt_row


    @torch.no_grad()
    def inference(
        self,
        mistake_audio,
        score_audio,
        audio_path=None,
        prompt_path=None,
        outpath=None,
        mistake_midi_path=None,
        valid_programs=None,
        num_beams=1,
        batch_size=1,
        max_length=1024,
        verbose=False,
    ):
        # print("== DEBUG: Entering inference ==")
        # print("self.model.config:", self.model.config)

        try:
            # Check valid programs
            if valid_programs is not None:
                invalid_programs = self._get_program_ids(valid_programs)
            else:
                invalid_programs = None

            mel_length = 256

            mistake_inputs, score_inputs, mistake_frame_times, score_frame_times, prompt_row = self._preprocess(
                mistake_audio, score_audio, prompt_path, mistake_midi_path=mistake_midi_path
            )

            mistake_inputs = mistake_inputs[:, :mel_length, :]
            score_inputs = score_inputs[:, :mel_length, :]
            mistake_inputs_tensor = torch.from_numpy(mistake_inputs)
            score_inputs_tensor = torch.from_numpy(score_inputs)

            # If prompts are enabled
            if self.model.config.use_prompt:
                prompt_row = self._postprocess_prompt_batch(prompt_row, dataset)
                # Combine prompt tokens
                stacked_prompt_tokens = torch.cat(prompt_row["prompt_tokens"], dim=0)
                stacked_prompt_masks = torch.cat(prompt_row["prompt_masks"], dim=0)
            else:
                stacked_prompt_tokens, stacked_prompt_masks = None, None

            # Batch everything for the model
            (
                mistake_batches,
                score_batches,
                mistake_frame_times_batch,
                score_frame_times_batch,
                prompt_tokens_batch,
                prompt_attention_mask_batch,
            ) = self._batching(
                mistake_inputs_tensor,
                score_inputs_tensor,
                mistake_frame_times,
                score_frame_times,
                stacked_prompt_tokens if self.model.config.use_prompt else None,
                stacked_prompt_masks if self.model.config.use_prompt else None,
                batch_size,
            )

            # Process each batch
            results = []
            for idx, (mistake_batch, score_batch, pt_batch, pt_mask_batch) in enumerate(
                zip(mistake_batches, score_batches, prompt_tokens_batch, prompt_attention_mask_batch)
            ):

                result = self.model.generate(
                    mistake_inputs=mistake_batch,
                    score_inputs=score_batch,
                    decoder_input_ids=pt_batch if pt_batch is not None else None,
                    decoder_attention_mask=pt_mask_batch if pt_mask_batch is not None else None,
                    max_length=max_length,
                    num_beams=num_beams,
                    do_sample=False,
                    length_penalty=0.4,
                    eos_token_id=self.model.config.eos_token_id,
                    early_stopping=False,
                    bad_words_ids=invalid_programs,
                    use_cache=False, # TODO: may need to change back to false
                )
                results.append(self._postprocess_batch(result))

            # Combine final results
            event = self._to_event(results, mistake_frame_times_batch)
            reference_ns = self._mistake_reference_ns or self._prompt_reference_ns
            if reference_ns is not None:
                # Copy tempo map and other global timing metadata from the
                # reference (prefer the mistake MIDI, fall back to the prompt)
                # so the rendered MIDI stays aligned with the intended tempo
                # grid.
                event.ClearField("tempos")
                for tempo in reference_ns.tempos:
                    event.tempos.add().CopyFrom(tempo)
                event.ClearField("time_signatures")
                for ts in reference_ns.time_signatures:
                    event.time_signatures.add().CopyFrom(ts)
                if reference_ns.ticks_per_quarter:
                    event.ticks_per_quarter = reference_ns.ticks_per_quarter
            if outpath is None:
                filename = audio_path.split("/")[-1].split(".")[0]
                outpath = f"./out/{filename}.mid"
            os.makedirs(os.path.dirname(outpath), exist_ok=True)
            note_seq.note_sequence_to_midi_file(event, outpath)

        except Exception as e:
            traceback.print_exc()
    def _postprocess_batch(self, result):
        after_eos = torch.cumsum(
            (result == self.model.config.eos_token_id).float(), dim=-1
        )
        # minus special token
        result = (
            result - self.vocab.num_special_tokens()
        )  # tokens are offset by special tokes.
        result = torch.where(
            after_eos.bool(), -1, result
        )  # mark tokens after EOS as -1 (invalid token)
        # remove bos (SOS token)
        result = result[:, 1:]
        result = result.cpu().numpy()
        return result

    def _to_event(self, predictions_np: List[np.ndarray], frame_times: np.ndarray):
        predictions = []
        for i, batch in enumerate(predictions_np):
            for j, tokens in enumerate(batch):
                tokens = tokens[
                    : np.argmax(tokens == vocabularies.DECODED_EOS_ID)
                ]  # trim after EOS
                start_time = frame_times[i][j][0]  # get start time of the frame
                start_time -= start_time % (
                    1 / self.codec.steps_per_second
                )  # rounding down time. Why?
                predictions.append(
                    {
                        "est_tokens": tokens,
                        "start_time": start_time,
                        "raw_inputs": [],
                    }  # raw_inputs is empty
                )

        encoding_spec = (
            note_sequences.NoteEncodingWithTiesSpec
        )  # here we use ties to tie seperate event frames together
        result = metrics_utils.event_predictions_to_ns(
            predictions, codec=self.codec, encoding_spec=encoding_spec
        )
        return result["est_ns"]



    
    
def save_frames(mistake_frames, score_frames, input_path, file_prefix='frames'):
    output_dir = os.path.dirname(input_path)
    
    num_frames = mistake_frames.shape[0]  # Assuming all frames have the same number
    
    for i in range(num_frames):
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        
        axs[0].imshow(mistake_frames[i,:,:].T, aspect='auto', origin='lower', cmap='viridis')
        axs[0].set_title('Mistake Frames')
        axs[0].set_xlabel('Frame Index')
        axs[0].set_ylabel('Frequency Bin')

        axs[1].imshow(score_frames[i,:,:].T, aspect='auto', origin='lower', cmap='viridis')
        axs[1].set_title('Score Frames _wo')
        axs[1].set_xlabel('Frame Index')
        axs[1].set_ylabel('Frequency Bin')

        plt.tight_layout()
        
        output_path = os.path.join(output_dir, f'{file_prefix}_frames_{i}.png')
        plt.savefig(output_path)
        
        plt.close() 
        
def print_audio_length(audio_frames, frame_rate):
    audio_length_seconds = len(audio_frames) / frame_rate
    print(f"Audio length: {audio_length_seconds} seconds", flush=True)
