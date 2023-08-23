import os
import torch
import glob

class SoundEventDetectionDataset(torch.utils.data.Dataset):

    def __init__(self, config, data_dir):
        pass

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class ActiveSoundEventDetectionDataset(torch.utils.data.Dataset):

    def __init__(self, config, data_dir):
        self.wav_paths = glob.glob(os.path.join(data_dir, "*.wav"))

        self.segment_size = config['segment_size']

        self.wav_path_annotations = {}
        for wav_path in wav_paths:
            self.wav_path_to_annotations[wav_path] = []

        # allocate arrays
        self.xs = np.zeros(config['max_nb_segments'], config['segment_size'])
        self.ys = np.zeros(config['max_nb_segments'])

        self.last_batch_size = 0
        self.total_segments  = 0

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def update(self, labels, wav_paths, timings):
        print("Updating active learning dataset ...")
        self.last_batch_size = 0
        for idx in range(len(labels)):
            wav_path               = wav_paths[idx]
            label                  = labels[idx]
            (start_time, end_time) = timings[idx]

            # keep track of annotated timings
            self.wav_path_to_annotations[wav_path].append((start_time, end_time, label))

            # load the actual data
            x_batch, y_batch = load_batch(wav_path, start_time, end_time, label, segment_size=self.segment_size)

            self.xs[self.total_segments:self.total_segments+len(x_batch), :] = x_batch
            self.ys[self.total_segments:self.total_segments+len(y_batch)] = y_batch

            # increment the last batch size
            self.last_batch_size += len(x_batch)
            # increment the total segment count
            self.total_segments += len(x_batch)

    def get_unlabeled_timings(self):
        return


#class SyntheticBioacousticDataset(torch.utils.data.Dataset):
#
#    def __init__(self, split=='train', config):
#
#        csv_paths = glob.glob(os.path.join(config['data_dir'], "*.csv"))
#
#        n_csvs = len(csv_paths)
#        train_paths, test_paths = csv_paths[:n_csvs//2], csv_paths[n_csvs//2:]
#
#        
#        if split == 'train':
#            self.x, self.y = self.load_data(train_paths)
#        else:
#            self.x, self.y = self.load_data(test_paths)
#
#    def load_data(self):
#        return
#
#    def __len__(self):
#        return len(self.x)
#
#    def __getitem__(self, idx):
#        return self.x[idx], self.y[idx]
