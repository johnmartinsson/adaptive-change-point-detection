import os

import glob
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
import argparse

import stats_utils
import sed_utils

def write_foreground_events_to_dir(wave_path, class_name, annotations, dir_path):
    wave, sample_rate = sf.read(wave_path)
    basename = os.path.basename(wave_path).replace('.wav', '')

    for idx, (start_time, end_time) in enumerate(annotations):
        wave_segment = wave[int(np.floor(start_time*sample_rate)):int(np.ceil(end_time*sample_rate))]

        class_dir = os.path.join(dir_path, class_name)

        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        sf.write(os.path.join(class_dir, "{}_{}.wav".format(basename, idx)), wave_segment, sample_rate)

def write_background_events_to_dir(wave_path, class_name, annotations, dir_path, min_duration=10):
    wave, sample_rate = sf.read(wave_path)
    basename = os.path.basename(wave_path).replace('.wav', '')

    counter = 0
    # each annotated event is used at least once
    while counter < len(annotations):
        duration = 0
        background_events = []
        while duration < min_duration:
            # cycle if not enough at the end
            idx = counter % len(annotations)
            start_time, end_time = annotations[idx]

            wave_segment = wave[int(np.floor(start_time*sample_rate)):int(np.ceil(end_time*sample_rate))]
            background_events.append(wave_segment)

            duration += (end_time - start_time)
            counter += 1

        class_dir = os.path.join(dir_path, class_name)

        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        background_segment = np.concatenate(background_events)

        sf.write(os.path.join(class_dir, "{}_{}.wav".format(basename, idx)), background_segment, sample_rate)

def dcase_to_scaper_source_material(csv_paths, fg_dir, bg_dir, class_name):
    for csv_path in csv_paths:
        n_shots = 1000000
        pos_anns = stats_utils.get_positive_annotations(csv_path, n_shots=n_shots, class_name=class_name, expand=0.0)
        neg_anns = stats_utils.get_gap_annotations(csv_path, n_shots=n_shots, class_name=class_name)

        assert(len(pos_anns) < n_shots)
        assert(len(neg_anns) < n_shots)

        wave_path = csv_path.replace('.csv', '.wav')
        write_foreground_events_to_dir(wave_path, 'fg', pos_anns, fg_dir)
        write_background_events_to_dir(wave_path, 'bg', neg_anns, bg_dir, min_duration=10)

def main():
    parser = argparse.ArgumentParser(description='A data conversion script.')
    parser.add_argument('--data_dir', help='The data dir containing (name.wav, name.csv) pairs', required=True, type=str)
    parser.add_argument('--scaper_dir', help='The data dir to write the foreground, background data', required=True, type=str)
    parser.add_argument('--class_name', help='The name of the positive sound event class (rest treated as negative)', required=True, type=str)
    args = parser.parse_args()

    # create scaper dirs
    fg_test_dir = os.path.join(args.scaper_dir, args.class_name, 'test_source', 'foreground')
    bg_test_dir = os.path.join(args.scaper_dir, args.class_name, 'test_source', 'background')
    fg_train_dir = os.path.join(args.scaper_dir, args.class_name, 'train_source', 'foreground')
    bg_train_dir = os.path.join(args.scaper_dir, args.class_name, 'train_source', 'background')

    # split, read, convert, and write data
    _csv_paths = glob.glob(os.path.join(args.data_dir, '*.csv'))

    csv_paths = []
    for csv_path in _csv_paths:
        df = pd.read_csv(csv_path)
        if args.class_name in df.columns:
            csv_paths.append(csv_path)
    print(csv_paths)

    assert(len(csv_paths) >= 2)


    n = len(csv_paths)
    train_csv_paths, test_csv_paths = csv_paths[n//2:], csv_paths[:n//2]

    dcase_to_scaper_source_material(train_csv_paths, fg_train_dir, bg_train_dir, args.class_name)
    dcase_to_scaper_source_material(test_csv_paths, fg_test_dir, bg_test_dir, args.class_name)
   
if __name__ == '__main__':
    main()
