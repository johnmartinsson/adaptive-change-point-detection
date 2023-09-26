import os

import glob
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
import argparse

import stats_utils
import sed_utils

def dcase_to_soundscape(csv_paths, soundscape_dir, class_name):
    if not os.path.exists(soundscape_dir):
        os.makedirs(soundscape_dir)

    for csv_path in csv_paths:
        n_shots = 1000000
        pos_anns = stats_utils.get_positive_annotations(csv_path, n_shots=n_shots, class_name=class_name, expand=0.0)

        assert(len(pos_anns) < n_shots)

        csv_basename = os.path.basename(csv_path)

        # write annotations to dir
        with open(os.path.join(soundscape_dir, csv_basename.replace('.csv', '.txt')), 'w') as fw:
            for (start_time, end_time) in pos_anns:
                l = "{}\t{}\t{}\n".format(start_time, end_time, 'fg')
                fw.write(l)

        # copy wav file to dir
        wave_path = csv_path.replace('.csv', '.wav')
        wave, sample_rate = sf.read(wave_path)
        sf.write(os.path.join(soundscape_dir, csv_basename.replace('.csv', '.wav')), wave, sample_rate)

def main():
    parser = argparse.ArgumentParser(description='A data conversion script.')
    parser.add_argument('--data_dir', help='The data dir containing (name.wav, name.csv) pairs', required=True, type=str)
    parser.add_argument('--soundscape_dir', help='The data dir to write the soundscapes', required=True, type=str)
    parser.add_argument('--class_name', help='The name of the positive sound event class (rest treated as negative)', required=True, type=str)
    args = parser.parse_args()

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
    print("train: ", train_csv_paths)
    print("test: ", test_csv_paths)

    dcase_to_soundscape(train_csv_paths, os.path.join(args.soundscape_dir, 'train'), args.class_name)
    dcase_to_soundscape(test_csv_paths,  os.path.join(args.soundscape_dir, 'test'), args.class_name)
   
if __name__ == '__main__':
    main()
