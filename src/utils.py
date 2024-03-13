import pandas as pd

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

def sort_by_rank(ranks, values, reverse=True):
    return [x[1] for x in sorted(list(zip(ranks, values)), key=lambda x: x[0], reverse=reverse)]

def print_queries(queries, name):
    print("{} : ".format(name), ['({:.2f}, {:.2f})'.format(c[0], c[1]) for c in queries])

def shift_bit_length(x):
    x = int(x)
    return 1<<(x-1).bit_length()

def next_power_of_2(x):
    return shift_bit_length(x)
