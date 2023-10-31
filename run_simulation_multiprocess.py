import os
import sys
import numpy as np
import tqdm
import copy
import glob
import time

import warnings

from multiprocessing import Pool
from functools import partial

import models
import datasets
import oracles
import metrics
import utils
import evaluate
import visualize

import matplotlib.pyplot as plt
import matplotlib

def list_difference(l1, l2):
    return sorted(list(set(l1).difference(l2)))

def simulate_strategy(query_strategy, soundscape_basenames, n_queries, base_dir, min_iou, noise_factor, normalize, iteration):
    next_soundscape_basename = query_strategy.next_soundscape_basename(soundscape_basenames)

    
    # evaluate label-quality and get embeddings
    f1_score, mean_iou_score, p_embeddings, n_embeddings = evaluate.evaluate_query_strategy(
        base_dir            = base_dir,
        soundscape_basename = next_soundscape_basename,
        query_strategy      = query_strategy,
        n_queries           = n_queries,
        min_iou             = min_iou,
        noise_factor        = noise_factor,
        normalize           = normalize,
        iteration           = iteration,
    )

    soundscape_basenames_remaining = list_difference(soundscape_basenames, [next_soundscape_basename])

    query_strategy.update(p_embeddings, n_embeddings)
    
    return f1_score, mean_iou_score, p_embeddings, n_embeddings, soundscape_basenames_remaining

# TODO: include this in default loop
def evaluate_annotation_process_on_test_data(query_strategy, base_dir, n_queries, noise_factor):
    soundscape_basenames = [os.path.basename(b).split('.')[0] for b in glob.glob(os.path.join(base_dir, "*.wav"))]

    f1s   = []
    mious = []

    oracle = oracles.WeakLabelOracle(base_dir)

    for soundscape_basename in soundscape_basenames:
        ref_pos  = datasets.load_pos_ref_aux(base_dir, soundscape_basename)

        queries = query_strategy.predict_queries(base_dir, soundscape_basename, n_queries, noise_factor=noise_factor)
        pred_pos = oracle.pos_events_from_queries(queries, soundscape_basename)

        if not len(pred_pos) == 0:
            f1   = metrics.f1_score_from_events(ref_pos, pred_pos, min_iou=0.00000001)
            miou = metrics.average_matched_iou(ref_pos, pred_pos, min_iou=0.00000001)
            f1s.append(f1)
            mious.append(miou)
        else:
            warnings.warn("No predictions, results will potentially be skewed ...")
            print("query strategy, fixed = {}, CPD = {}".format(query_strategy.fixed_queries, query_strategy.emb_cpd))
            print("pos_pred: ", pred_pos)
            print("pos_ref: ", ref_pos)
            print("queries: ", queries)
            # TODO: not sure, strong penalization of no predictions
            f1s.append(0)
            mious.append(0)
            
    return np.mean(f1s), np.mean(mious)



def evaluate_model_on_test_data(query_strategy, base_dir, threshold=0.5):
    soundscape_basenames = [os.path.basename(b).split('.')[0] for b in glob.glob(os.path.join(base_dir, "*.wav"))]

    f1s   = []
    mious = []
    for soundscape_basename in soundscape_basenames:
        ref_pos  = datasets.load_pos_ref_aux(base_dir, soundscape_basename)
        pred_pos = query_strategy.predict_pos_events(base_dir, soundscape_basename, threshold=threshold)
        if not len(pred_pos) == 0:
            f1   = metrics.f1_score_from_events(ref_pos, pred_pos, min_iou=0.00000001)
            miou = metrics.average_matched_iou(ref_pos, pred_pos, min_iou=0.00000001)
            f1s.append(f1)
            mious.append(miou)
        else:
            # TODO: not sure, strong penalization of no predictions
            f1s.append(0)
            mious.append(0)
            
    return np.mean(f1s), np.mean(mious)

def run_annotation_simulation(idx_init, base_dir, all_soundscape_basenames, n_soundscapes_budget, idx_query_strategy, test_base_dir, n_queries, min_iou, noise_factor, normalize):
    # state
    # all_soundscape_basenames
    init_soundscape_basename = all_soundscape_basenames[idx_init]


    # remove initial soundscape
    # TODO: this is not strictly needed anymore, since we no longer initialize with soundscapes
    remaining_soundscape_basenames = list_difference(all_soundscape_basenames, [init_soundscape_basename])
    remaining_soundscape_basenames = sorted(remaining_soundscape_basenames)

    # query strategies
    # TODO: actually make a parent class QueryStrategy, and then create FixedQueryStrategy, AdaptiveQueryStrategy, CPDQueryStrategy
    if idx_query_strategy == 0:
        query_strategy = models.AdaptiveQueryStrategy(base_dir, random_soundscape=False,  fixed_queries=False, normal_prototypes=True)
    elif idx_query_strategy == 1:
        query_strategy = models.AdaptiveQueryStrategy(base_dir, random_soundscape=True,  fixed_queries=False, normal_prototypes=True)
    elif idx_query_strategy == 2:
        query_strategy = models.AdaptiveQueryStrategy(base_dir, random_soundscape=True, fixed_queries=True, emb_cpd=True, normal_prototypes=True)
    elif idx_query_strategy == 3:
        query_strategy = models.AdaptiveQueryStrategy(base_dir, random_soundscape=True,  fixed_queries=True, normal_prototypes=True)
    else:
        raise ValueError("not defined ..")
    
    # initialize strategies with ground truth labels for supplied soundscape
    # TODO: this may actually not be necessary ...
    #query_strategy.initialize_with_ground_truth_labels(init_soundscape_basename)

    bns = copy.copy(remaining_soundscape_basenames)
    
    # label the soundscapes using budget
    f1_scores_test    = np.zeros(n_soundscapes_budget)
    miou_scores_test  = np.zeros(n_soundscapes_budget)
    f1_scores_train   = np.zeros(n_soundscapes_budget)
    miou_scores_train = np.zeros(n_soundscapes_budget)

    budget_count = 0
    while budget_count < n_soundscapes_budget:
        f1_test_score, miou_test_score = evaluate_model_on_test_data(query_strategy, test_base_dir)

        f1_scores_test[budget_count]   = f1_test_score
        miou_scores_test[budget_count] = miou_test_score

        f1_train_score, miou_train_score, _, _, bns = simulate_strategy(
            query_strategy       = query_strategy,
            soundscape_basenames = bns,
            n_queries            = n_queries,
            base_dir             = base_dir,
            min_iou              = min_iou,
            noise_factor         = noise_factor,
            normalize            = normalize,
            iteration            = budget_count,
        )

        f1_scores_train[budget_count]   = f1_train_score
        miou_scores_train[budget_count] = miou_train_score

        # increase budget count
        budget_count += 1

    return f1_scores_test, miou_scores_test, f1_scores_train, miou_scores_train


def main():
    sim_dir = sys.argv[1]
    print(sim_dir)

    if not os.path.exists(sim_dir):
        os.makedirs(sim_dir)

    snr = '0.0'
    n_soundscapes = 100
    
    n_queriess = [7, 10, 20, 30, 40] #[7, 10, 20, 30, 40, 50]
    
    base_dir      = '/mnt/storage_1/datasets/bioacoustic_sed/generated_datasets/me_0.8s_0.25s_large_final/train_soundscapes_snr_{}/'.format(snr)
    test_base_dir = '/mnt/storage_1/datasets/bioacoustic_sed/generated_datasets/me_0.8s_0.25s_large_final/test_soundscapes_snr_{}/'.format(snr)
    #test_soundscape_basename = 'soundscape_0'
    
    all_soundscape_basenames = ['soundscape_{}'.format(idx) for idx in range(n_soundscapes)]
    
    
    n_soundscapes_budget = n_soundscapes-1
    min_iou = 0.00001
    
    n_init_soundscapes = 10
    normalize_embeddings = True

    # only use these strategies
    # TODO: using only CPD baseline strategy now
    query_strategy_indices = [1,2,3]

    assert(n_soundscapes_budget < n_soundscapes)
    assert(n_init_soundscapes <= n_soundscapes)
    
    print("n_queriess: ", n_queriess)

    f1_scores_test     = np.zeros((4, len(n_queriess), n_init_soundscapes, n_soundscapes_budget))
    miou_scores_test   = np.zeros((4, len(n_queriess), n_init_soundscapes, n_soundscapes_budget))
    f1_scores_train    = np.zeros((4, len(n_queriess), n_init_soundscapes, n_soundscapes_budget))
    miou_scores_train  = np.zeros((4, len(n_queriess), n_init_soundscapes, n_soundscapes_budget))
    
    total_time = time.time()
    for idx_n_queries, n_queries in enumerate(tqdm.tqdm(n_queriess)):
        init_indices = np.random.choice(np.arange(len(all_soundscape_basenames)), n_init_soundscapes)
        for idx_query_strategy in query_strategy_indices:
            with Pool(8) as p:
                f = partial(run_annotation_simulation,
                        # bind state to function
                        base_dir                 = base_dir,
                        all_soundscape_basenames = all_soundscape_basenames,
                        n_soundscapes_budget     = n_soundscapes_budget,
                        idx_query_strategy       = idx_query_strategy,
                        test_base_dir            = test_base_dir,
                        n_queries                = n_queries,
                        min_iou                  = min_iou,
                        noise_factor             = 0,
                        normalize                = normalize_embeddings,
                )
                res = p.map(f, init_indices)
                res = np.array(list(zip(*res)))
                test_f1    = res[0]
                test_miou  = res[1] 
                train_f1   = res[2]
                train_miou = res[3]

                f1_scores_test[idx_query_strategy, idx_n_queries, :, :] = test_f1
                miou_scores_test[idx_query_strategy, idx_n_queries, :, :] = test_miou

                f1_scores_train[idx_query_strategy, idx_n_queries, :, :] = train_f1
                miou_scores_train[idx_query_strategy, idx_n_queries, :, :] = train_miou
           
    np.save(os.path.join(sim_dir, "f1_scores_train.npy"), f1_scores_train)
    np.save(os.path.join(sim_dir, "miou_scores_train.npy"), miou_scores_train)
    np.save(os.path.join(sim_dir, "f1_scores_test.npy"), f1_scores_test)
    np.save(os.path.join(sim_dir, "miou_scores_test.npy"), miou_scores_test)

    print("total time: ", time.time()-total_time)

if __name__ == '__main__':
    main()
