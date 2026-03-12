"""
Multi-track transcription evaluation script for dataset.
"""
import os
import mir_eval
import glob
import pretty_midi
import numpy as np
import librosa
import note_seq
import collections
import concurrent.futures
import traceback
from tqdm import tqdm
import tempfile


# def get_granular_program(program_number, is_drum, granularity_type):
#     """
#     Returns the granular program number based on the given parameters.

#     Parameters:
#     program_number (int): The original program number.
#     is_drum (bool): Indicates whether the program is a drum program or not.
#     granularity_type (str): The type of granularity to apply.

#     Returns:
#     int: The granular program number.

#     """
#     if granularity_type == "full":
#         return program_number
#     elif granularity_type == "midi_class":
#         return (program_number // 8) * 8
#     elif granularity_type == "flat":
#         return 0 if not is_drum else 1


# Standard evaluation Pipeline for MT3
def compute_transcription_metrics(ref_mid, est_mid):
    """Helper function to compute onset/offset, onset only, and frame metrics."""
    ns_ref = note_seq.midi_file_to_note_sequence(ref_mid)
    ns_est = note_seq.midi_file_to_note_sequence(est_mid)
    intervals_ref, pitches_ref, _ = note_seq.sequences_lib.sequence_to_valued_intervals(
        ns_ref
    )
    intervals_est, pitches_est, _ = note_seq.sequences_lib.sequence_to_valued_intervals(
        ns_est
    )
    len_est_intervals = len(intervals_est)
    len_ref_intervals = len(intervals_ref)

    # onset-offset
    onoff_precision, onoff_recall, onoff_f1, onoff_overlap = (
        mir_eval.transcription.precision_recall_f1_overlap(
            intervals_ref, pitches_ref, intervals_est, pitches_est
        )
    )

    # onset-only
    on_precision, on_recall, on_f1, on_overlap = (
        mir_eval.transcription.precision_recall_f1_overlap(
            intervals_ref, pitches_ref, intervals_est, pitches_est, offset_ratio=None
        )
    )

    return {
        "len_ref_intervals": len_ref_intervals,
        "len_est_intervals": len_est_intervals,
        "onoff_precision": onoff_precision,
        "onoff_recall": onoff_recall,
        "onoff_f1": onoff_f1,
        "onoff_overlap": onoff_overlap,
        "on_precision": on_precision,
        "on_recall": on_recall,
        "on_f1": on_f1,
        "on_overlap": on_overlap,
    }


# This is multi-instrument F1 score
def mt3_program_aware_note_scores(fextra, fremoved, fmistakes, est, granularity_type):
    """
    Edited version of MT3's program aware precision/recall/F1 score.
    We follow Perceiver's evaluation approach which takes only onset and program into account.
    Using MIDIs transcribed from MT3, we managed to get similar results as Perceiver, which is 0.75 for onset F1.
    """
    # ref_extra_mid = pretty_midi.PrettyMIDI(fextra) # reference midi (music score)
    # ref_removed_mid = pretty_midi.PrettyMIDI(fremoved) # reference midi (music score)
    # ref_mistakes_mid = pretty_midi.PrettyMIDI(fmistakes) # reference midi (music score)   
    # List of MIDI files to combine
    midi_files = [fextra, fremoved, fmistakes]

    # Create a new PrettyMIDI object for the combined MIDI
    combined_midi = pretty_midi.PrettyMIDI()

    # Load each file and add its instruments to the combined MIDI
    for midi_file in midi_files:
        # Load the current MIDI file
        current_midi = pretty_midi.PrettyMIDI(midi_file)
        
        # Add each instrument from the current file to the combined MIDI file
        for instrument in current_midi.instruments:
            combined_midi.instruments.append(instrument)   
    ref_mid = combined_midi # reference midi (music score) 
    est_mid = pretty_midi.PrettyMIDI(est) # estimated midi (transcription)
    # Print details about each instrument (treated as a track)
    for index, instrument in enumerate(est_mid.instruments):
        is_drum = 'Drum' if instrument.is_drum else 'Not Drum'
        instrument_name = pretty_midi.program_to_instrument_name(instrument.program)
        # print(f"Track {index}: Instrument {instrument.program}, Name: {instrument_name}, {is_drum}")
        
    # Write the combined MIDI to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mid') as temp_midi_file:
        combined_midi.write(temp_midi_file.name)
        temp_midi_file_path = temp_midi_file.name

    res = dict() # results
    ref_ns = note_seq.midi_file_to_note_sequence(temp_midi_file_path)
    
    est_ns = note_seq.midi_file_to_note_sequence(est)
    # TODO: We might need to remove drums and process separately as in MT3
    # NOTE: We don't need to remove drums and process separately as in MT3
    # as we consider onset only for all instruments.
    # def remove_drums(ns):
    #   ns_drumless = note_seq.NoteSequence()
    #   ns_drumless.CopyFrom(ns)
    #   del ns_drumless.notes[:]
    #   ns_drumless.notes.extend([note for note in ns.notes if not note.is_drum])
    #   return ns_drumless

    # est_ns_drumless = remove_drums(est_ns)
    # ref_ns_drumless = remove_drums(ref_ns)

    est_tracks = [est_ns]
    ref_tracks = [ref_ns]
    use_track_offsets = [False]
    use_track_velocities = [False]
    track_instrument_names = [""]
    

    for est_ns, ref_ns, use_offsets, use_velocities, instrument_name in zip(
        est_tracks,
        ref_tracks,
        use_track_offsets,
        use_track_velocities,
        track_instrument_names,
    ):

        est_intervals, est_pitches, est_velocities = (
            note_seq.sequences_lib.sequence_to_valued_intervals(est_ns)
        )
        # intervals is like (note.start_time, note.end_time)

        ref_intervals, ref_pitches, ref_velocities = (
            note_seq.sequences_lib.sequence_to_valued_intervals(ref_ns)
        )

        # Precision / recall / F1 using onsets (and pitches) only.
        # looks like we can just do this seperately for each type of error!
        precision, recall, f_measure, avg_overlap_ratio = (
            mir_eval.transcription.precision_recall_f1_overlap(
                ref_intervals=ref_intervals,
                ref_pitches=ref_pitches,
                est_intervals=est_intervals,
                est_pitches=est_pitches,
                offset_ratio=None,
            )
        )
        res["Onset precision"] = precision
        res["Onset recall"] = recall
        res["Onset F1"] = f_measure
        
        
        
    # Iterate over each estimated track and corresponding reference track
    for track_index, (ref_mid, est_instrument) in enumerate(zip(midi_files, est_mid.instruments)):
        # Skip if estimated track has more instruments than the reference tracks
        

        
        ref_ns = note_seq.midi_file_to_note_sequence(ref_mid)

        # Write the estimated instrument to a temporary file
        with tempfile.NamedTemporaryFile(delete=True, suffix='.mid') as est_temp_midi_file:
            est_temp_mid = pretty_midi.PrettyMIDI()
            est_temp_mid.instruments.append(est_instrument)
            est_temp_mid.write(est_temp_midi_file.name)
            est_ns = note_seq.midi_file_to_note_sequence(est_temp_midi_file.name)

        # Calculate evaluation metrics
        est_intervals, est_pitches, _ = (
            note_seq.sequences_lib.sequence_to_valued_intervals(est_ns)
        )
        ref_intervals, ref_pitches, _ = (
            note_seq.sequences_lib.sequence_to_valued_intervals(ref_ns)
        )

        precision, recall, f_measure, _ = mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals=ref_intervals,
            ref_pitches=ref_pitches,
            est_intervals=est_intervals,
            est_pitches=est_pitches,
            offset_ratio=None
        )

        # Store results for this track
        res[f"Track {track_index} precision"] = precision
        res[f"Track {track_index} recall"] = recall
        res[f"Track {track_index} F1"] = f_measure

   
    return res


# def loop_transcription_eval(ref_mid, est_mid):
#     """
#     This evaluation takes in account the separability of the model. Goes by "track" instead of tight
#     coupling to "program number". This is because of a few reasons:
#     - for loops, the program number in ref can be arbitrary
#         - e.g. how do you assign program number to Vox?
#         - no one use program number for synth / sampler etc.
#         - string contrabass VS bass midi class are different, but can be acceptable
#         - leads and key / synth pads and electric piano
#     - the "track splitting" aspect is more important than the accuracy of the midi program number
#         - we can have wrong program number, but as long as they are grouped in the correct track
#     - hence we propose 2 more evaluation metrics:
#         - f1_score_matrix for each ref_track VS est_track, take the mean of the maximum f1 score for each ref_track
#         - number of tracks
#     """
#     score_matrix = np.zeros((len(ref_mid.instruments), len(est_mid.instruments)))

#     for i, ref_inst in enumerate(ref_mid.instruments):
#         for j, est_inst in enumerate(est_mid.instruments):
#             if ref_inst.is_drum == est_inst.is_drum:
#                 ref_intervals = np.array(
#                     [[note.start, note.end] for note in ref_inst.notes]
#                 )
#                 ref_pitches = np.array(
#                     [librosa.midi_to_hz(note.pitch) for note in ref_inst.notes]
#                 )
#                 est_intervals = np.array(
#                     [[note.start, note.end] for note in est_inst.notes]
#                 )
#                 est_pitches = np.array(
#                     [librosa.midi_to_hz(note.pitch) for note in est_inst.notes]
#                 )

#                 _, _, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
#                     ref_intervals, ref_pitches, est_intervals, est_pitches
#                 )
#                 score_matrix[i][j] = f1

#     inst_idx = np.argmax(score_matrix, axis=-1)
#     ref_progs = [inst.program for inst in ref_mid.instruments]
#     est_progs = [est_mid.instruments[idx].program for idx in inst_idx]
#     return (
#         np.mean(np.max(score_matrix, axis=-1)),
#         len(ref_mid.instruments),
#         len(est_mid.instruments),
#     )

# Called by test, TODO: see how to use this.
def evaluate_main(
    dataset_name,  # "MAESTRO" or "CocoChorales"
    test_midi_dir,
    ground_truth,
    enable_instrument_eval=False, # TODO: check what this means
    first_n=None,
    estimate_filename="mix.mid",
):
    if dataset_name in ["MAESTRO", "CocoChorales"]:
        # ------------------------------------------------------------------
        # 1. Gather all inferred MIDI estimates in the evaluation directory
        # ------------------------------------------------------------------
        pattern = os.path.join(test_midi_dir, "*", estimate_filename)
        estimate_paths = glob.glob(pattern)
        if not estimate_paths and estimate_filename != "mix.mid":
            # Fall back to legacy naming to keep compatibility when mixing caches
            estimate_paths = glob.glob(os.path.join(test_midi_dir, "*", "mix.mid"))
        estimate_map = {os.path.basename(os.path.dirname(p)): p for p in estimate_paths}

        # Map track_ids to various midi_paths from ground_truth
        track_to_extra = {gt["track_id"]: gt["extra_notes_midi"] for gt in ground_truth}
        track_to_removed = {gt["track_id"]: gt["removed_notes_midi"] for gt in ground_truth}
        track_to_mistake = {gt["track_id"]: gt["correct_notes_midi"] for gt in ground_truth}

        # ------------------------------------------------------------------
        # 2. Build reference‑to‑estimate tuples **by track_id**, never by list index
        # ------------------------------------------------------------------
        track_ids = sorted(track_to_extra.keys())          # same order every run
        fnames = []                                        # (extra, removed, mistake, est)

        missing_estimates = []
        for tid in track_ids:
            if tid in estimate_map:
                fnames.append((
                    track_to_extra[tid],
                    track_to_removed[tid],
                    track_to_mistake[tid],
                    estimate_map[tid],
                ))
            else:
                missing_estimates.append(tid)

        extra_estimates = [tid for tid in estimate_map if tid not in track_to_extra]

        # ------------------------------------------------------------------
        # 3. Diagnostics
        # ------------------------------------------------------------------
        if extra_estimates:
            print(f"[WARN] {len(extra_estimates)} estimate files without ground‑truth:", extra_estimates[:10])
        if missing_estimates:
            print(f"[WARN] {len(missing_estimates)} ground‑truth items without estimates:", missing_estimates[:10])

        print(f"[INFO] Aligned pairs: {len(fnames)}  |  estimates found: {len(estimate_paths)}  |  ground‑truth entries: {len(track_ids)}")
        if not fnames:
            raise ValueError("No aligned estimate/ground‑truth pairs found.  Check the cache path or dataset mapping.")

        # ------------------------------------------------------------------
        # 4. Optional subset for quick sanity checks
        # ------------------------------------------------------------------
        if first_n:
            fnames = fnames[:first_n]
    
    else:
        raise ValueError("dataset_name must be either MAESTRO or CocoChorales.")

    def func(item):
        fextra, fremoved, fmistake, est = item

        results = {}
        # The granularity loop seems to be redundant as mt3_program_aware_note_scores doesn't use it.
        # We run it once.
        dic = mt3_program_aware_note_scores(fextra, fremoved, fmistake, est, "flat")
        results.update(dic)

        return results

    pbar = tqdm(total=len(fnames), desc="Evaluating MIDI files")
    scores = collections.defaultdict(list)
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        future_to_fname = {executor.submit(func, fname): fname for fname in fnames}
        for future in concurrent.futures.as_completed(future_to_fname):
            try:
                dic = future.result()
                for item in dic:
                    scores[item].append(dic[item])
                pbar.update(1)

                # Update TQDM postfix with running averages for key metrics
                postfix_stats = {}
                if "Onset F1" in scores and scores["Onset F1"]:
                    postfix_stats["Onset F1"] = f"{np.mean(scores['Onset F1']):.4f}"
                if "Onset precision" in scores and scores["Onset precision"]:
                    postfix_stats["P"] = f"{np.mean(scores['Onset precision']):.3f}"
                if "Onset recall" in scores and scores["Onset recall"]:
                    postfix_stats["R"] = f"{np.mean(scores['Onset recall']):.3f}"
                if postfix_stats:
                    pbar.set_postfix(postfix_stats)

            except Exception:
                failed_item = future_to_fname[future]
                print(f"\nError processing file triplet: {failed_item}. Skipping.")
                traceback.print_exc()

    pbar.close()
    # The verbose print of the entire scores dictionary is removed for clarity.
    # print(f"scores: {scores}")
    print("\n--- Final Mean Scores ---")
    mean_scores = {k: np.mean(v) for k, v in scores.items() if k != "F1 by program" and v}

    if enable_instrument_eval:
        print("====")
        program_f1_dict = {}
        # Ensure 'F1 by program' exists and is not empty before processing
        if "F1 by program" in scores and scores["F1 by program"]:
            for item in scores["F1 by program"]:
                for key in item:
                    if key not in program_f1_dict:
                        program_f1_dict[key] = []
                    program_f1_dict[key].append(item[key])
            
            program_f1_dict = {k: np.mean(np.array(v)) for k, v in program_f1_dict.items()}

            d = {
                -1: "Drums", 0: "Piano", 1: "Chromatic Percussion", 2: "Organ",
                3: "Guitar", 4: "Bass", 5: "Strings", 6: "Ensemble",
                7: "Brass", 8: "Reed", 9: "Pipe", 10: "Synth Lead",
                11: "Synth Pad", 12: "Synth Effects",
            }

            for key in sorted(d.keys()):
                if key == -1:
                    if key in program_f1_dict:
                        print("{}: {:.4f}".format(d[key], program_f1_dict[key]))
                elif key * 8 in program_f1_dict:
                    print("{}: {:.4f}".format(d[key], program_f1_dict[key * 8]))
        else:
            print("No 'F1 by program' scores to evaluate.")

    return scores, mean_scores
