import sys
import os
import sys
import numpy as np
import tqdm
import copy
import glob
import time

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

def simulate_strategy(query_strategy, soundscape_basenames, n_queries, base_dir, min_iou):
    next_soundscape_basename = query_strategy.next_soundscape_basename(soundscape_basenames)

    
    # evaluate label-quality and get embeddings
    f1_score, mean_iou_score, p_embeddings, n_embeddings = evaluate.evaluate_query_strategy(
        base_dir            = base_dir,
        soundscape_basename = next_soundscape_basename,
        query_strategy      = query_strategy,
        n_queries           = n_queries,
        min_iou             = min_iou
    )

    soundscape_basenames_remaining = list_difference(soundscape_basenames, [next_soundscape_basename])

    query_strategy.update(p_embeddings, n_embeddings)
    
    return f1_score, mean_iou_score, p_embeddings, n_embeddings, soundscape_basenames_remaining

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

def main():
    sim_dir = sys.argv[1]

    if not os.path.exists(sim_dir):
        os.makedirs(sim_dir)

    snr = '0.0'
    n_soundscapes = 100
    
    n_queriess = [7, 10, 30, 50]
    print("n_queriess: ", n_queriess)
    
    base_dir      = '/mnt/storage_1/datasets/bioacoustic_sed/generated_datasets/me_0.8s_0.25s_large/train_soundscapes_snr_{}/'.format(snr)
    test_base_dir = '/mnt/storage_1/datasets/bioacoustic_sed/generated_datasets/me_0.8s_0.25s_large/test_soundscapes_snr_{}/'.format(snr)
    test_soundscape_basename = 'soundscape_0'
    
    all_soundscape_basenames = ['soundscape_{}'.format(idx) for idx in range(n_soundscapes)]
    
    
    n_soundscapes_budget = 40
    min_iou = 0.00001
    
    n_init_soundscapes = 10

    # only use these strategies
    query_strategy_indices = [1, 3]

    assert(n_soundscapes_budget < n_soundscapes)
    assert(n_init_soundscapes <= n_soundscapes)
    
    f1_scores_test     = np.zeros((4, len(n_queriess), n_init_soundscapes, n_soundscapes_budget))
    miou_scores_test   = np.zeros((4, len(n_queriess), n_init_soundscapes, n_soundscapes_budget))
    f1_scores_train    = np.zeros((4, len(n_queriess), n_init_soundscapes, n_soundscapes_budget))
    miou_scores_train  = np.zeros((4, len(n_queriess), n_init_soundscapes, n_soundscapes_budget))
    
    total_time = time.time()
    for idx_n_queries, n_queries in enumerate(n_queriess):
        init_soundscape_basenames = np.random.choice(['soundscape_{}'.format(idx) for idx in range(n_soundscapes)], n_init_soundscapes)
        
        # parallelize over this
        for idx_init, init_soundscape_basename in enumerate(tqdm.tqdm((init_soundscape_basenames))):
            # remove initial soundscape
            remaining_soundscape_basenames = list_difference(all_soundscape_basenames, [init_soundscape_basename])
            remaining_soundscape_basenames = sorted(remaining_soundscape_basenames)
            
            # query strategies
            query_strategy_0 = models.AdaptiveQueryStrategy(base_dir, random_soundscape=False, fixed_queries=False)
            query_strategy_1 = models.AdaptiveQueryStrategy(base_dir, random_soundscape=True,  fixed_queries=False)
            query_strategy_2 = models.AdaptiveQueryStrategy(base_dir, random_soundscape=False, fixed_queries=True)
            query_strategy_3 = models.AdaptiveQueryStrategy(base_dir, random_soundscape=True,  fixed_queries=True)
            
            # initialize strategies with ground truth labels for supplied soundscape
            query_strategy_0.initialize_with_ground_truth_labels(init_soundscape_basename)
            query_strategy_1.initialize_with_ground_truth_labels(init_soundscape_basename)
            query_strategy_2.initialize_with_ground_truth_labels(init_soundscape_basename)
            query_strategy_3.initialize_with_ground_truth_labels(init_soundscape_basename)
    
            query_strategies = [None, query_strategy_1, None, query_strategy_3]
    
            bnss = []
            for _ in range(len(query_strategies)):
                bnss.append(copy.copy(remaining_soundscape_basenames))
            
            # label the soundscapes using budget
            budget_count = 0
            while budget_count < n_soundscapes_budget:
    
                #print("-------------------------------------------------------------------------")
                # simulate annotation of one soundscape for each query strategy
                for idx_query_strategy in query_strategy_indices:
                    query_strategy = query_strategies[idx_query_strategy]
    
                    #t1 = time.time()
                    f1_test_score, miou_test_score = evaluate_model_on_test_data(query_strategy, test_base_dir)
                    #print("evaluation time: ", time.time() - t1)
                    
                    f1_scores_test[idx_query_strategy, idx_n_queries, idx_init, budget_count] = f1_test_score
                    miou_scores_test[idx_query_strategy, idx_n_queries, idx_init, budget_count] = miou_test_score
   
                    bns = bnss[idx_query_strategy]
                    #t1 = time.time()
                    f1_train_score, miou_train_score, _, _, bns = simulate_strategy(
                        query_strategy       = query_strategy,
                        soundscape_basenames = bns,
                        n_queries            = n_queries,
                        base_dir             = base_dir,
                        min_iou              = min_iou
                    )
                    #print("simulation time: ", time.time() - t1)
                    bnss[idx_query_strategy] = bns
                    #print("strategy {}: ".format(idx_query_strategy), bns)
    
    
                    f1_scores_train[idx_query_strategy, idx_n_queries, idx_init, budget_count] = f1_train_score
                    miou_scores_train[idx_query_strategy, idx_n_queries, idx_init, budget_count] = miou_train_score
    
    
                # increase budget count
                budget_count += 1
        
        print("------------------------------------------------")
        print("- Number of queries: {}".format(n_queries))
        print("------------------------------------------------")
        for idx_query_strategy in range(len(query_strategies)):
            f1_mean_train = f1_scores_train[idx_query_strategy, idx_n_queries].flatten().mean()
            f1_std_train  = f1_scores_train[idx_query_strategy, idx_n_queries].flatten().std()
    
            miou_mean_train = miou_scores_train[idx_query_strategy, idx_n_queries].flatten().mean()
            miou_std_train  = miou_scores_train[idx_query_strategy, idx_n_queries].flatten().std()
            print("Strategy {}, f1 = {:.3f} +- {:.3f}, miou = {:.3f} +- {:.3f}".format(idx_query_strategy, f1_mean_train, f1_std_train, miou_mean_train, miou_std_train))

    
    np.save(os.path.join(sim_dir, "f1_scores_train.npy"), f1_scores_train)
    np.save(os.path.join(sim_dir, "miou_scores_train.npy"), miou_scores_train)
    np.save(os.path.join(sim_dir, "f1_scores_test.npy"), f1_scores_test)
    np.save(os.path.join(sim_dir, "miou_scores_test.npy"), miou_scores_test)

    print("total time: ", time.time()-total_time)

if __name__ == '__main__':
    main()
