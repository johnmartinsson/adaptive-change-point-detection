import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import scipy
import librosa
import tqdm
import copy
import glob

import models
import datasets
import oracles
import metrics
import utils
import change_point_detection as cpd
import query_strategies as qs
import evaluate
import visualize

import metrics

import matplotlib.pyplot as plt
import matplotlib

import argparse

def list_difference(l1, l2):
    return sorted(list(set(l1).difference(l2)))

def simulate_strategy(query_strategy, soundscape_basenames, n_queries, base_dir, min_iou, noise_factor, normalize, iteration, emb_win_length, fp_noise, fn_noise):
    # get the next soundscape to annotate
    next_soundscape_basename = query_strategy.next_soundscape_basename(soundscape_basenames)

    # evaluate label-quality and get embeddings and annotations
    f1_score, mean_iou_score, p_embeddings, n_embeddings, pos_pred = evaluate.evaluate_query_strategy( # TODO: this should probably be moved to the query strategy class
        base_dir            = base_dir,
        soundscape_basename = next_soundscape_basename,
        query_strategy      = query_strategy,
        n_queries           = n_queries,
        min_iou             = min_iou,
        noise_factor        = noise_factor,
        normalize           = normalize,
        iteration           = iteration,
        emb_win_length      = emb_win_length,
        fp_noise            = fp_noise,
        fn_noise            = fn_noise
    )

    soundscape_basenames_remaining = list_difference(soundscape_basenames, [next_soundscape_basename])

    query_strategy.update(p_embeddings, n_embeddings)

    soundscape_preds = (next_soundscape_basename,  pos_pred)
    
    return f1_score, mean_iou_score, p_embeddings, n_embeddings, soundscape_basenames_remaining, soundscape_preds




colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', help='The directory to save the results in', required=True, type=str)
    parser.add_argument('--name', required=True, type=str)
    parser.add_argument('--n_soundscapes_budget', required=True, type=int)
    parser.add_argument('--n_queries_budget', required=True, type=int)
    parser.add_argument('--class_name', required=True, type=str)
    #parser.add_argument('--n_eval_freq', required=True, type=int)
    parser.add_argument('--n_runs', required=True, type=int)
    parser.add_argument('--emb_win_length', required=True, type=float)
    parser.add_argument('--fp_noise', required=True, type=float)
    parser.add_argument('--fn_noise', required=True, type=float)

    args = parser.parse_args()

    # The data directories
    emb_win_length = args.emb_win_length
    emb_hop_length = emb_win_length / 4

    emb_hop_length_str = '{:.2f}'.format(emb_hop_length)
    emb_win_length_str = '{:.1f}'.format(emb_win_length)
    class_name = args.class_name

    base_dir =     train_base_dir = '/mnt/storage_1/datasets/bioacoustic_sed/generated_datasets/{}_{}_{}s/train_soundscapes_snr_0.0'.format(class_name, emb_win_length_str, emb_hop_length_str)
    
    #base_dir      = args.train_data_dir
    test_base_dir = base_dir.replace('train', 'test')
    print("base_dir: ", base_dir)
    print("test_base_dir: ", test_base_dir)

    n_soundscapes        = args.n_soundscapes_budget
    n_soundscapes_budget = args.n_soundscapes_budget
    n_queries            = args.n_queries_budget
    #n_eval               = args.n_eval_freq
    n_runs               = args.n_runs

    # Gaussian initialization of prototypes
    normal_prototypes    = True
    # Zero mean unit variance normalization of embeddings
    normalize_embeddings = True


    # Names of the strategies (plots and tables)
    strategy_names = ['OPT', 'ADP', 'CPD', 'FIX']

    # Strategies to actually run
    indices_query_strategies = [0, 1, 2, 3]

    min_iou = 0.000001
    noise_factor = 0
    test_soundscape_basename = 'soundscape_0'


    f1_scores_test     = np.zeros((4, n_runs, 1, n_soundscapes_budget))
    miou_scores_test   = np.zeros((4, n_runs, 1, n_soundscapes_budget))
    f1_scores_train    = np.zeros((4, n_runs, 1, n_soundscapes_budget))
    miou_scores_train  = np.zeros((4, n_runs, 1, n_soundscapes_budget))


    f1_scores_train_online   = np.zeros((4, n_runs, 1, n_soundscapes_budget))
    miou_scores_train_online = np.zeros((4, n_runs, 1, n_soundscapes_budget))

    print("normal prototypes    = ", normal_prototypes)
    print("normalize embeddings = ", normalize_embeddings)

    #dx_n_queries = 0
    # TODO: legacy, used to initialize with an annotated soundscape, no longer required
    idx_init = 0

    for idx_run in range(n_runs):
        # initalize methods

        opt_query_strategy = models.AdaptiveQueryStrategy(base_dir, random_soundscape=True,  fixed_queries=False, opt_queries=True,    normal_prototypes=normal_prototypes)
        adp_query_strategy = models.AdaptiveQueryStrategy(base_dir, random_soundscape=True,  fixed_queries=False, emb_cpd=False, normal_prototypes=normal_prototypes)
        cpd_query_strategy = models.AdaptiveQueryStrategy(base_dir, random_soundscape=True,  fixed_queries=True,  emb_cpd=True,  normal_prototypes=normal_prototypes)
        fix_query_strategy = models.AdaptiveQueryStrategy(base_dir, random_soundscape=True,  fixed_queries=True,  emb_cpd=False, normal_prototypes=normal_prototypes)
        query_strategies = [opt_query_strategy, adp_query_strategy, cpd_query_strategy, fix_query_strategy]
        
        # initial unlabeled soundscapes
        all_soundscape_basenames = ['soundscape_{}'.format(idx) for idx in range(n_soundscapes)]
        remaining_soundscape_basenames = all_soundscape_basenames
        remaining_soundscape_basenames = sorted(remaining_soundscape_basenames)

        bnss = []
        for _ in range(len(query_strategies)):
            bnss.append(copy.copy(remaining_soundscape_basenames))  
      
        budget_count = 0
        print("####################################################")
        print("Run: {}".format(idx_run))
        print("####################################################")

        while budget_count < n_soundscapes_budget:
            print("----------------- {} -----------------------".format(budget_count))
            #print("iteration {}".format(budget_count))
            for idx_query_strategy in indices_query_strategies: 
                query_strategy = query_strategies[idx_query_strategy]

                # if False: #budget_count % n_eval == 0:
                #     f1_test_score, miou_test_score   = evaluate_model_on_test_data(query_strategy, test_base_dir)
                #     f1_train_score, miou_train_score = evaluate_annotation_process_on_test_data(query_strategy, test_base_dir, n_queries, noise_factor, args.fp_noise, args.fn_noise)

                #     f1_scores_train[idx_query_strategy, idx_run, idx_init, budget_count//n_eval] = f1_train_score
                #     miou_scores_train[idx_query_strategy, idx_run, idx_init, budget_count//n_eval] = miou_train_score

                #     f1_scores_test[idx_query_strategy, idx_run, idx_init, budget_count//n_eval] = f1_test_score
                #     miou_scores_test[idx_query_strategy, idx_run, idx_init, budget_count//n_eval] = miou_test_score

                #     #print("-------------------------------------")
                #     print("strategy {}, iteration {}, f1 = {:.3f}, miou = {:.3f} (train)".format(strategy_names[idx_query_strategy], budget_count, f1_train_score, miou_train_score))
                #     print("strategy {}, iteration {}, f1 = {:.3f}, miou = {:.3f} (test)".format(strategy_names[idx_query_strategy], budget_count, f1_test_score, miou_test_score))

                bns = bnss[idx_query_strategy]

                f1_train, miou_train, _, _, bns, soundscape_preds = simulate_strategy(
                    query_strategy       = query_strategy,
                    soundscape_basenames = bns,
                    n_queries            = n_queries,
                    base_dir             = base_dir,
                    min_iou              = min_iou,
                    noise_factor         = noise_factor,
                    normalize            = normalize_embeddings,
                    iteration            = budget_count,
                    emb_win_length       = args.emb_win_length,
                    fp_noise             = args.fp_noise,
                    fn_noise             = args.fn_noise,
                )

                # retreive and store annotations to disk
                # results/<experiment_name>/<method_name>/<run>/train_annotations/<iteration_filename>.tsv  # the training annotations

                annotated_soundscape_basename, annotations = soundscape_preds
                query_strategy_name = strategy_names[idx_query_strategy]
                train_annotation_dir = os.path.join(args.results_dir, args.name, query_strategy_name, str(idx_run), 'train_annotations')
                if not os.path.exists(train_annotation_dir):
                    os.makedirs(train_annotation_dir)

                # save annotations in .tsv format
                with open(os.path.join(train_annotation_dir, "iter_{}_".format(budget_count) + annotated_soundscape_basename + '.tsv'), 'w') as f:
                    f.write('onset\toffset\tevent_label\n')
                    for (onset, offset) in annotations:
                        f.write('{}\t{}\t{}\n'.format(onset, offset, args.class_name))

                # save annotations in .npy format
                #np.save(os.path.join(train_annotation_dir, "iter_{}_".format(budget_count) + annotated_soundscape_basename + '.npy'), np.array(annotations))

                f1_scores_train_online[idx_query_strategy, idx_run, idx_init, budget_count]   = f1_train
                miou_scores_train_online[idx_query_strategy, idx_run, idx_init, budget_count] = miou_train

                #print("simulation time: ", time.time() - t1)
                bnss[idx_query_strategy] = bns
                #print("strategy {}: ".format(idx_query_strategy), bns)

                #print("strategy {}, iteration {}, f1 = {:.2f}, miou = {:.2f} (train)".format(idx_query_strategy, budget_count, f1_train_score, miou_train_score))

            # increase budget count
            budget_count += 1

    print("done! saving results in {} ...".format(os.path.join(args.results_dir, args.name)))
    # save evaluation
    # np.save(os.path.join(args.results_dir, args.name, "f1_scores_train.npy"), f1_scores_train)
    # np.save(os.path.join(args.results_dir, args.name, "miou_scores_train.npy"), miou_scores_train)
    # np.save(os.path.join(args.results_dir, args.name, "f1_scores_test.npy"), f1_scores_test)
    # np.save(os.path.join(args.results_dir, args.name, "miou_scores_test.npy"), miou_scores_test)

    # shape (n_query_strategies, n_runs, 1, n_soundscapes_budget)
    np.save(os.path.join(args.results_dir, args.name, "f1_scores_train_online.npy"), f1_scores_train_online)
    np.save(os.path.join(args.results_dir, args.name, "miou_scores_train_online.npy"), miou_scores_train_online)

if __name__ == '__main__':
    main()
