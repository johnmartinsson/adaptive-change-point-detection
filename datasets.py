import os
import glob
import numpy as np

def load_timings_and_embeddings(file_path, embedding_dim=1024):
    timings_and_embeddings = []
    with open(file_path, 'r') as f:
        data = f.readlines()
        for d in data:
            _d = d.split('\t')
            start_time = _d[0]
            end_time = _d[1]
            embedding = np.zeros(embedding_dim)
            values = _d[2].split(',')
            assert(len(embedding) == len(values))
            for i, value in enumerate(values):
                embedding[i] = float(value)
            timings_and_embeddings.append((float(start_time), float(end_time), embedding))
    return timings_and_embeddings

class FixedQueryActiveLearningDataset:
    def __init__(self, base_dir):
        super(FixedQueryActiveLearningDataset, self).__init__()
        embedding_file_paths = glob.glob(os.path.join(base_dir, '*.embeddings.txt'))

        if len(embedding_file_paths) == 0:
            raise ValueError("no embeddings files found: {}".format(base_dir))

        self.train_data = {}
        self.labeled_train_data = {}
        
        # Thought: this becomes tricky for dynamic timings
        for embedding_file_path in embedding_file_paths:
            timings_and_embeddings = load_timings_and_embeddings(embedding_file_path)
            self.train_data[os.path.basename(embedding_file_path).split('.')[0]] = timings_and_embeddings
            self.labeled_train_data[os.path.basename(embedding_file_path).split('.')[0]] = np.zeros(len(timings_and_embeddings))-1

    def count_labeled(self):
        count = 0
        for key in self.labeled_train_data.keys():
            indices = np.where(self.labeled_train_data[key] >= 0)[0]
            count += len(indices)
        return count

    def count_unlabeled(self):
        count = 0
        for key in self.labeled_train_data.keys():
            indices = np.where(self.labeled_train_data[key] == -1)[0]
            count += len(indices)
        return count

    def add_query_label(self, key, idx, label):
        self.labeled_train_data[key][idx] = label
    
    def random_query(self):
        queries = []
        for key in self.labeled_train_data.keys():
            indices = np.where(self.labeled_train_data[key] == -1)
            for idx in indices[0]:
                queries.append((key, idx))

        query_idx = np.random.randint(len(queries))
        (key, idx) = queries[query_idx]
        
        return (key, idx), self.train_data[key][idx]

class AdaptiveQueryActiveLearningDataset:
    def __init__(self, base_dir):
        raise ValueError("not yey implemented ...")
        embedding_file_paths = glob.glob(os.path.join(base_dir, '*.embeddings.txt'))
