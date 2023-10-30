import numpy as np
import os
import librosa

import datasets
import change_point_detection as cpd

def get_soundscape_length(base_dir, soundscape_name):

    wav_path = os.path.join(base_dir, '{}.wav'.format(soundscape_name))
    wave, sr = librosa.load(wav_path, sr=None)
    soundscape_length = len(wave) / sr

    return soundscape_length

def fixed_query_strategy(n_queries, base_dir, soundscape_name, soundscape_length):
    fix_queries = np.linspace(0, soundscape_length, n_queries+1)
    fix_queries = list(zip(fix_queries[:-1], fix_queries[1:]))
    
    return fix_queries

def load_query_ref(ref_path, soundscape_length):
    pos_ref = datasets.load_pos_ref(ref_path)
    neg_ref = datasets.load_neg_ref(ref_path, soundscape_length)

    # sort by start time
    query_ref = sorted(pos_ref + neg_ref, key=lambda x: x[0])

    return query_ref

def optimal_query_strategy(base_dir, soundscape_name, soundscape_length):
    ref_path = os.path.join(base_dir, '{}.txt'.format(soundscape_name))

    opt_queries = load_query_ref(ref_path, soundscape_length)

    return opt_queries

def change_point_query_strategy(n_queries, base_dir, soundscape_name, soundscape_length, normalize=False):
    timings, embeddings = datasets.load_timings_and_embeddings(
        base_dir = base_dir,
        soundscape_basename = soundscape_name,
        embedding_dim=1024,
        normalize=normalize
    )
    
    # extract change-points
    peak_timings, ds = cpd.change_point_detection_from_embeddings(
        embeddings,
        timings,
        M = 1,
        prominence = 0,
        n_peaks = n_queries-1
    )

    # create change-point queries
    query_starts  = [0] + peak_timings
    query_ends = peak_timings + [soundscape_length]
    cp_queries = list(zip(query_starts, query_ends))

    return cp_queries
