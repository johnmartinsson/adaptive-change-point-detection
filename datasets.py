import os
import glob
import numpy as np

def load_pos_ref_aux(base_dir, soundscape_basename):
    ref_path = os.path.join(base_dir, "{}.txt".format(soundscape_basename))
    return load_pos_ref(ref_path)

def load_pos_ref(ref_path):
    pos_ref = []
    
    with open(ref_path, 'r') as f:
        ls = f.readlines()
        for l in ls:
            ds = l.split('\t')
            start_time = float(ds[0])
            end_time = float(ds[1])
            pos_ref.append((start_time, end_time))
    return pos_ref

def compute_neg_from_pos(pos_ref, soundscape_length):
    if len(pos_ref) == 0:
        return [(0, soundscape_length)]
    pos_ref = sorted(pos_ref, key=lambda x: x[0])

    neg_ref = []
    prev_pos_end_time   = 0
    for (curr_pos_start_time, curr_pos_end_time) in pos_ref:

        neg_start_time = prev_pos_end_time
        neg_end_time   = curr_pos_start_time

        if curr_pos_start_time >= prev_pos_end_time:
            assert neg_start_time <= neg_end_time, "start time less than end time ..."
            neg_ref.append((neg_start_time, neg_end_time))
    
        prev_pos_end_time   = curr_pos_end_time
    
    end_times = [e for (_, e) in pos_ref]
    latest_end_time = np.max(end_times)

    if latest_end_time < soundscape_length:
        neg_ref.append((latest_end_time, soundscape_length))

    return neg_ref

   
def load_neg_ref(ref_path, soundscape_length):
    pos_ref = load_pos_ref(ref_path)
    neg_ref = compute_neg_from_pos(pos_ref, soundscape_length)
    return neg_ref

def normalize_embeddings(embeddings, base_dir):
    if 'test' in base_dir:
        _base_dir = base_dir.replace('test', 'train')
    else:
        _base_dir = base_dir

    embedding_mean_path = os.path.join(_base_dir, 'embedding_mean.npy')
    embedding_std_path  = os.path.join(_base_dir, 'embedding_std.npy')

    if not os.path.exists(embedding_mean_path):
        print("computing mean and std for data in: ", _base_dir)
        soundscape_basenames = glob.glob(os.path.join(_base_dir, '*.birdnet.embeddings.txt'))
        soundscape_basenames = [os.path.basename(s).split('.')[0] for s in soundscape_basenames]
        train_embeddingss = []
        for soundscape_basename in soundscape_basenames:
            assert not 'test' in _base_dir, "normalizing using the test data!"
            _, train_embeddings = load_timings_and_embeddings(_base_dir, soundscape_basename, normalize=False)

            #print(train_embeddings.shape)

            train_embeddingss.append(train_embeddings)
        train_embeddingss = np.array(train_embeddingss)

        #print(train_embeddingss.shape)
        mean = np.mean(train_embeddingss, axis=(0, 1))
        std  = np.std(train_embeddingss, axis=(0, 1))

        # save mean and std
        print("save mean and std")
        print(embedding_mean_path)
        print(embedding_std_path)
        np.save(embedding_mean_path, mean)
        np.save(embedding_std_path, std)

    mean = np.load(embedding_mean_path)
    std  = np.load(embedding_std_path)

    # normalize
    # TODO: how to handle?
    std[std == 0] = 1
    embeddings = embeddings - mean
    embeddings = embeddings / std #(std + 1e-10)

    return embeddings

def load_timings_and_embeddings(base_dir, soundscape_basename, embedding_dim=1024, normalize=True):
    file_path = os.path.join(base_dir, '{}.birdnet.embeddings.txt'.format(soundscape_basename))
    timings = []
    embeddings = []
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
            timings.append((float(start_time), float(end_time)))
            embeddings.append(embedding)

    embeddings = np.array(embeddings)

    if normalize:
        #print("normalizing embeddings ...")
        embeddings = normalize_embeddings(embeddings, base_dir)

    end_times = np.array([e for (_, e) in timings])
    indices = end_times <= 30.0
    timings = np.array(timings)[indices]
    embeddings = embeddings[indices]

    return timings, embeddings