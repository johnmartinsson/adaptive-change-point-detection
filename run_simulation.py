import os
import sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import copy

import models
import evaluate

import argparse



def list_difference(l1, l2):
    return sorted(list(set(l1).difference(l2)))

def simulate_strategy(query_strategy, soundscape_basenames, n_queries, base_dir, min_iou, noise_factor, normalize, iteration, emb_win_length, fp_noise, fn_noise):
    # get the next soundscape to annotate
    next_soundscape_basename = query_strategy.next_soundscape_basename(soundscape_basenames)

    # evaluate label-quality and get embeddings and annotations
    f1_score, mean_iou_score, p_embeddings, n_embeddings, pos_pred, used_queries = evaluate.evaluate_query_strategy( # TODO: this should probably be moved to the query strategy class
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
    
    return f1_score, mean_iou_score, p_embeddings, n_embeddings, soundscape_basenames_remaining, soundscape_preds, used_queries

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', help='The directory to save the results in', required=True, type=str)
    parser.add_argument('--n_soundscapes_budget', required=True, type=int)
    parser.add_argument('--n_queries_budget', required=True, type=int)
    parser.add_argument('--class_name', required=True, type=str)
    parser.add_argument('--n_runs', required=True, type=int)
    parser.add_argument('--emb_win_length', required=True, type=float)
    parser.add_argument('--fp_noise', required=True, type=float)
    parser.add_argument('--fn_noise', required=True, type=float)
    parser.add_argument('--base_dir', required=True, type=str)
    #parser.add_argument('--prominence_threshold', required=True, type=float)
    #parser.add_argument('--coverage_threshold', required=True, type=float)

    args = parser.parse_args()

    # The data directories
    emb_win_length = args.emb_win_length
    emb_hop_length = emb_win_length / 4

    emb_hop_length_str = '{:.2f}'.format(emb_hop_length)
    emb_win_length_str = '{:.1f}'.format(emb_win_length)
    class_name = args.class_name

    base_dir = '{}/generated_datasets/{}_{}_{}s/train_soundscapes_snr_0.0'.format(args.base_dir, class_name, emb_win_length_str, emb_hop_length_str)
    
    # test_base_dir = base_dir.replace('train', 'test')
    # print("base_dir: ", base_dir)
    # print("test_base_dir: ", test_base_dir)

    n_soundscapes        = args.n_soundscapes_budget
    n_soundscapes_budget = args.n_soundscapes_budget
    n_queries            = args.n_queries_budget
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

    f1_scores_train_online   = np.zeros((4, n_runs, 1, n_soundscapes_budget))
    miou_scores_train_online = np.zeros((4, n_runs, 1, n_soundscapes_budget))

    query_count = np.zeros((4, n_runs))

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
        # print("####################################################")
        # print("Run: {}".format(idx_run))
        # print("####################################################")

        while budget_count < n_soundscapes_budget:
            sys.stdout.write("Class name: {}, run: {}/{}, annotated soundscapes: {}/{}   \r".format(class_name, idx_run+1, n_runs, budget_count, n_soundscapes_budget))
            sys.stdout.flush()
            #print("----------------- {} -----------------------".format(budget_count))
            for idx_query_strategy in indices_query_strategies: 
                query_strategy = query_strategies[idx_query_strategy]

                bns = bnss[idx_query_strategy]

                f1_train, miou_train, _, _, bns, soundscape_preds, used_queries = simulate_strategy(
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

                query_count[idx_query_strategy, idx_run] += used_queries

                annotated_soundscape_basename, annotations = soundscape_preds
                query_strategy_name = strategy_names[idx_query_strategy]
                settings_str = "n_queries_{}_noise_{}".format(args.n_queries_budget, args.fn_noise)
                train_annotation_dir = os.path.join(args.results_dir, args.class_name, settings_str, query_strategy_name, str(idx_run), 'train_annotations')
                if not os.path.exists(train_annotation_dir):
                    os.makedirs(train_annotation_dir)

                # save annotations in .tsv format
                with open(os.path.join(train_annotation_dir, "iter_{}_".format(budget_count) + annotated_soundscape_basename + '.tsv'), 'w') as f:
                    f.write('onset\toffset\tevent_label\n')
                    for (onset, offset) in annotations:
                        f.write('{}\t{}\t{}\n'.format(onset, offset, args.class_name))

                f1_scores_train_online[idx_query_strategy, idx_run, idx_init, budget_count]   = f1_train
                miou_scores_train_online[idx_query_strategy, idx_run, idx_init, budget_count] = miou_train
                
                bnss[idx_query_strategy] = bns

            # increase budget count
            budget_count += 1

    #print("done! saving results in {} ...".format(os.path.join(args.results_dir, args.class_name)))

    # shape (n_query_strategies, n_runs, 1, n_soundscapes_budget)
    np.save(os.path.join(args.results_dir, args.class_name, settings_str, "f1_scores_train_online.npy"), f1_scores_train_online)
    np.save(os.path.join(args.results_dir, args.class_name, settings_str, "miou_scores_train_online.npy"), miou_scores_train_online)
    np.save(os.path.join(args.results_dir, args.class_name, settings_str, "query_count.npy"), query_count)

if __name__ == '__main__':
    main()
