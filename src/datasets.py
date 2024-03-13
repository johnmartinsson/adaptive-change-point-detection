import sys
import os
import glob
import numpy as np
import query_strategies as qs
import predict

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

        # normalize to zero mean unit variance
        mean = np.mean(train_embeddingss, axis=(0, 1))
        std  = np.std(train_embeddingss, axis=(0, 1))

        # save mean and std
        print("save mean and std")
        print(embedding_mean_path)
        print(embedding_std_path)
        np.save(embedding_mean_path, mean)
        np.save(embedding_std_path, std)

    mean = np.load(embedding_mean_path)
    #print("mean: ", mean.shape)
    std  = np.load(embedding_std_path)
    #print("std: ", std.shape)

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

    # TODO: changed to work with other than 30s soundscapes
    soundscape_length = qs.get_soundscape_length(base_dir, soundscape_basename)
    indices = end_times <= soundscape_length #30.0

    timings = np.array(timings)[indices]
    embeddings = embeddings[indices]

    return timings, embeddings


def load_annotated_embeddings(conf, sim_dir):
    # 1. load query strategy with given sample budget as in evaluation
    emb_win_length = conf.emb_win_length
    class_name     = conf.class_name
    idx_run        = 0
    strategy_name  = conf.strategy_name
    model_name     = conf.model_name
    base_dir       = conf.base_dir

    emb_win_length = emb_win_length
    emb_hop_length = emb_win_length / 4
    emb_hop_length_str = '{:.2f}'.format(emb_hop_length)
    emb_win_length_str = '{:.1f}'.format(emb_win_length)

    train_base_dir = conf.train_base_dir #os.path.join(base_dir, 'generated_datasets', '{}_{}_{}s'.format(class_name, emb_win_length_str, emb_hop_length_str), 'train_soundscapes_snr_0.0')

    train_annotation_dir   = os.path.join(sim_dir, str(idx_run), 'train_annotations')
    print(train_annotation_dir)

    # load train annotations
    train_annotation_paths = glob.glob(os.path.join(train_annotation_dir, "*.tsv"))

    def get_iteration(fp):
        return int(os.path.basename(fp).split('_')[1])

    def get_soundscape_basename(fp):
        return "_".join(os.path.basename(fp).split('_')[2:]).split('.')[0]

    evaluation_budget = 1.0

    # TODO: something goes wrong here with budget 1.0, max over empty list
    # the problem is most likely solved, the n_runs were inconsistent with the number of runs in the sim_dir
    n_soundscapes      = np.max([get_iteration(fp) for fp in train_annotation_paths]) + 1
    n_iter = int(evaluation_budget * n_soundscapes)

    sys.stdout.write("strategy = {}, run = {}, model_name = {}, budget = {}, n_iter = {}\n".format(strategy_name, idx_run, model_name, evaluation_budget, n_iter))

    # 1. load the annotations until n_iter
    budget_train_annotation_paths = [fp for fp in train_annotation_paths if get_iteration(fp) < n_iter]
    assert len(budget_train_annotation_paths) == n_iter, "budget not respected, expected {}, got {}".format(n_iter, len(budget_train_annotation_paths))
    soundscape_basenames          = [get_soundscape_basename(fp) for fp in budget_train_annotation_paths]

    # 2. load embeddings and annotations for the evaluation budget
    p_embss   = []
    n_embss   = []
    for idx, soundscape_basename in enumerate(soundscape_basenames):

        pos_ann = predict.get_positive_annotations(budget_train_annotation_paths[idx])
        p_embs, n_embs, _ = predict.get_embeddings_3(pos_ann, train_base_dir, soundscape_basename, emb_win_length)
        p_embs = np.array(p_embs)
        n_embs = np.array(n_embs)

        p_embss.append(p_embs)
        n_embss.append(n_embs)

    # positive and negative embeddings
    p_embs = np.concatenate(p_embss)
    n_embs = np.concatenate(n_embss)

    return p_embs, n_embs
