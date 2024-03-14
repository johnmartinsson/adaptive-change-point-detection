import sys
import os
sys.path.append(os.path.abspath('../'))

import utils

import glob
import argparse

import numpy as np
import soundfile as sf
import shutil
import yaml
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

def read_nigens_annotations(ann_path):
    with open(ann_path, 'r') as f:
        lines = f.readlines()
        anns = []
        for line in lines:
            s, e = line.split('\t')
            s = float(s)
            e = float(e)
            anns.append((s, e))
    return anns

def is_train(base_dir, wav_path):
    train_folds = ['NIGENS_8-foldSplit_fold{}.flist'.format(i) for i in [1,2,3,4,5,6]]
    test_folds = ['NIGENS_8-foldSplit_fold{}.flist'.format(i) for i in [7,8]]
    
    train_basenames = []
    test_basenames = []
    
    for train_fold in train_folds:
        with open(os.path.join(base_dir, train_fold), 'r') as f:
            ls = f.readlines()
            for l in ls:
                train_basenames.append(os.path.basename(l.strip()))
    for test_fold in test_folds:
        with open(os.path.join(base_dir, test_fold), 'r') as f:
            ls = f.readlines()
            for l in ls:
                test_basenames.append(os.path.basename(l.strip()))

    return os.path.basename(wav_path) in train_basenames

def produce_nigens_foreground_source_material(nigens_base_dir, source_base_dir, nigens_classes):
    for class_dir in nigens_classes:
        class_event_counter = 0
        plt.figure()

        wav_paths = glob.glob(os.path.join(nigens_base_dir, class_dir, '*.wav'))

        ls = []
        for wav_path in wav_paths:
            ann_path = wav_path + ".txt"
            anns = read_nigens_annotations(ann_path)
            wav, sr = sf.read(wav_path) #, sr=None)
            
            if is_train(nigens_base_dir, wav_path):
                split_dir = 'train_sources'
            else:
                split_dir = 'test_sources'
                
            fg_dir = 'foreground'
            
            absolute_class_dir = os.path.join(source_base_dir, split_dir, fg_dir, class_dir)
            if not os.path.exists(absolute_class_dir):
                os.makedirs(absolute_class_dir)
            
            basename = os.path.basename(wav_path).split('.')[0]
            
            for (s, e) in anns:
                wav_segment = wav[int(s*sr):int(e*sr)]
                print(wav_segment.shape, sr, len(wav_segment)/sr, e-s)
                class_event_counter += 1
                
                segment_path = '{}_{}.wav'.format(basename, class_event_counter)
                absolute_segment_path = os.path.join(absolute_class_dir, segment_path)
                print(absolute_segment_path)
                sf.write(absolute_segment_path, wav_segment, sr, subtype='PCM_16')
                
            
            _ls = [e-s for (s, e) in anns]
            ls.append(_ls)
            
        ls = np.concatenate(ls)
        sns.histplot(ls)
        plt.xlabel('event length [s]')
        plt.title(class_dir)

def produce_tut_background_source_material(tut_base_dir, source_base_dir):
    train_dir = 'train_sources'
    test_dir = 'test_sources'
    bg_dir = 'background'
    
    # TUT rare sounds 2017
    tut_bg_dir   = 'bgs'
    tut_splits_dir = 'cv_setup'

    bgs_devtest_yaml     = 'bgs_devtest.yaml'
    bgs_devtrain_yaml    = 'bgs_devtrain.yaml'

    splits = [bgs_devtest_yaml, bgs_devtrain_yaml]
    for split in splits:
        if 'train' in split:
            output_source_dir = train_dir
        else:
            output_source_dir = test_dir
            
        with open(os.path.join(tut_base_dir, tut_splits_dir, split), 'r') as f:
            meta_data = yaml.safe_load(f)

        for m in meta_data:
            class_name     = m['classname']
            class_name     = "_".join(class_name.split('/'))
            tut_audio_path = m['filepath']

            move_to_dir  = os.path.join(source_base_dir, output_source_dir, bg_dir, class_name)
            move_to_path = os.path.join(move_to_dir, os.path.basename(tut_audio_path))

            if not os.path.exists(move_to_dir):
                os.makedirs(move_to_dir)

            # move the audio to the correct directory
            move_from_path = os.path.join(tut_base_dir, tut_bg_dir, tut_audio_path)
            print("move_from_path: {}, move_to_path: {}".format(move_from_path, move_to_path))
            shutil.copyfile(move_from_path, move_to_path)

def write_foreground_events_to_dir(wave_path, annotations, source_base_dir, class_name):
    wave, sample_rate = sf.read(wave_path)
    basename = os.path.basename(wave_path).replace('.wav', '')

    print("wave_path: {}, sample_rate: {}, basename: {}".format(wave_path, sample_rate, basename))
    print("annotations: ", annotations)

    split_dirs = ['train_sources' for _ in range(len(annotations)//2)]
    split_dirs += ['test_sources' for _ in range(len(annotations)//2)]
    assert len(split_dirs) == len(annotations), "len(split_dirs): {}, len(annotations): {}".format(len(split_dirs), len(annotations))

    # random permutation with fixed seed for repdocibility
    split_dirs = np.random.default_rng(seed=42).permutation(split_dirs)

    for idx, (start_time, end_time) in enumerate(annotations):
        wave_segment = wave[int(np.floor(start_time*sample_rate)):int(np.ceil(end_time*sample_rate))]

        # fix random seed so reproducible
        split_dir = split_dirs[idx]
        class_dir = os.path.join(source_base_dir, split_dir, 'foreground', class_name)
        print("class_dir: ", class_dir)

        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        sf.write(os.path.join(class_dir, "{}_{}.wav".format(basename, idx)), wave_segment, sample_rate)

def dcase_to_scaper_source_material(csv_paths, source_base_dir, class_name):
    for csv_path in csv_paths:
        n_shots = 1000000
        pos_anns = utils.get_positive_annotations(csv_path, n_shots=n_shots, class_name='Q', expand=0.0)

        assert(len(pos_anns) < n_shots)

        wave_path = csv_path.replace('.csv', '.wav')
        write_foreground_events_to_dir(wave_path, pos_anns, source_base_dir, class_name)

def produce_dcase_foreground_source_material(dcase_base_dir, source_base_dir):
    # split, read, convert, and write data
    csv_paths = glob.glob(os.path.join(dcase_base_dir, '*.csv'))

    #print(_csv_paths)

    class_name = 'me'
    print("class_name: {}, csv_paths: {}".format(class_name, csv_paths))
    dcase_to_scaper_source_material(csv_paths, source_base_dir, class_name)

def main():
    parser = argparse.ArgumentParser(description='A data conversion script.')
    #parser.add_argument('--dcase_base_dir',  type=str, help='Path to the DCASE directory.')
    #parser.add_argument('--nigens_base_dir', type=str, help='Path to the NIGENS directory.')
    #parser.add_argument('--tut_base_dir',    type=str, help='Path to the TUT directory.')
    parser.add_argument('--source_base_dir', type=str, help='Path to the output source directory.')
    args = parser.parse_args()

    nigens_classes = ['dog', 'baby']
    dcase_classes = ['me']

    # TODO: download the data

    # Example: of the directory structure
    tut_base_dir    = '/mnt/storage_1/datasets/TUT_rare_sed_2017/TUT-rare-sound-events-2017-development/data/source_data/'
    nigens_base_dir = '/mnt/storage_1/datasets/NIGENS/'
    dcase_base_dir  = '/mnt/storage_1/datasets/bioacoustics_dcase2022/Development_Set/Validation_Set/ME/'
    
    #produce_dcase_foreground_source_material(dcase_base_dir, args.source_base_dir)
    produce_nigens_foreground_source_material(nigens_base_dir, args.source_base_dir, nigens_classes)
    produce_tut_background_source_material(tut_base_dir, args.source_base_dir)
    
    return

if __name__ == '__main__':
    main()