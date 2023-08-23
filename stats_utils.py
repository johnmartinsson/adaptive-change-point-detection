import numpy as np
import pandas as pd
import sed_utils

def get_positive_annotations(csv_path, n_shots, class_name, expand=0.0):
    ann_df = pd.read_csv(csv_path)
    ann_df = ann_df.sort_values('Starttime')
    ref_pos_indexes = ann_df.index[ann_df[class_name] == 'POS'].tolist()
    ann_nshot_df = ann_df.loc[ref_pos_indexes[:n_shots]]

    start_times = ann_nshot_df['Starttime'].tolist()
    start_times = [start_time - expand for start_time in start_times]

    end_times = ann_nshot_df['Endtime'].tolist()
    end_times = [end_time + expand for end_time in end_times]

    return list(zip(start_times, end_times))

def get_gap_annotations(csv_path, n_shots, class_name):
    pos_annotations = get_positive_annotations(csv_path, n_shots, class_name)
    start_times, end_times = list(zip(*pos_annotations))
    # a bit confusing, but true
    _end_times   = start_times[1:]
    _start_times = end_times[:-1]
 
    return list(zip(_start_times, _end_times))

def get_query_annotations(csv_path, n_shots, class_name):
    pos_annotations = get_positive_annotations(csv_path, n_shots, class_name)
    #start_time = pos_annotations[-1][1]
    start_time = 0
    wav_path = csv_path.replace('.csv', '.wav')
    wave, sample_rate = sed_utils.load_wave(wav_path)
    end_time = len(wave) / sample_rate
    return [(start_time, end_time)]

def get_event_lengths(n_shots, csv_path, class_name):
    ann_df = pd.read_csv(csv_path)
    ann_df = ann_df.sort_values('Starttime')
    ref_pos_indexes = ann_df.index[ann_df[class_name] == 'POS'].tolist()
    ann_nshot_df = ann_df.loc[ref_pos_indexes[:n_shots]]
    
    event_lengths = (ann_nshot_df['Endtime']-ann_nshot_df['Starttime'])
    
    return event_lengths

def get_density(n_shots, csv_path, class_name):
    ann_df = pd.read_csv(csv_path)
    ann_df = ann_df.sort_values('Starttime')
    ref_pos_indexes = ann_df.index[ann_df[class_name] == 'POS'].tolist()
    ann_nshot_df = ann_df.loc[ref_pos_indexes[:n_shots]]

    event_lengths = get_event_lengths(n_shots, csv_path, class_name)
    
    starttime = ann_nshot_df['Starttime'].iloc[0]
    endtime   = ann_nshot_df['Endtime'].iloc[-1]
    density   = event_lengths.sum() / (endtime-starttime)

    return density

def get_gap_lengths(n_shots, csv_path, class_name):
    ann_df = pd.read_csv(csv_path)
    ann_df = ann_df.sort_values('Starttime')
    ref_pos_indexes = ann_df.index[ann_df[class_name] == 'POS'].tolist()
    ann_5shot_df = ann_df.loc[ref_pos_indexes[:n_shots]]

    gap_lengths = (ann_5shot_df['Starttime'][1:].to_numpy()-ann_5shot_df['Endtime'][:-1].to_numpy())
    
    return gap_lengths

# backwards compability
def get_nshot_event_lengths(n_shots, csv_path):
    return get_event_lengths(n_shots, csv_path, 'Q')

def get_nshot_gap_lengths(n_shots, csv_path):
    return get_gap_lengths(n_shots, csv_path, 'Q')

def get_nshot_density(n_shots, csv_path):
    return get_density(n_shots, csv_path, 'Q')
