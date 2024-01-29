import os
import sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import copy
import glob

import models
import evaluate
import sound_event_eval

import argparse
import config



def list_difference(l1, l2):
    return sorted(list(set(l1).difference(l2)))

def simulate_strategy(query_strategy, soundscape_basenames, n_queries, base_dir, min_iou, noise_factor, normalize, iteration, emb_win_length, fp_noise, fn_noise, prominence_threshold, converage_threshold):
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
        fn_noise            = fn_noise,
        prominence_threshold = prominence_threshold,
        converage_threshold  = converage_threshold,
    )

    soundscape_basenames_remaining = list_difference(soundscape_basenames, [next_soundscape_basename])

    query_strategy.update(p_embeddings, n_embeddings)

    soundscape_preds = (next_soundscape_basename,  pos_pred)
    
    return f1_score, mean_iou_score, p_embeddings, n_embeddings, soundscape_basenames_remaining, soundscape_preds, used_queries

def run(conf):
    # Configuration file
    # conf = config.Config()

    ################################################################################
    # Simulate the active learning process
    ################################################################################

    # TODO: check all runs, and only start from the ones that are missing ...
    if not len(glob.glob(os.path.join(conf.sim_dir, '0', 'train_annotations', '*.tsv'))) == conf.n_soundscapes:
        print("Simulating active learning process ...")
        # TODO: move into own method
        f1_scores_train_online   = np.zeros((conf.n_runs, conf.n_soundscapes))
        miou_scores_train_online = np.zeros((conf.n_runs, conf.n_soundscapes))
        query_count = np.zeros((conf.n_runs))

        for idx_run in range(conf.n_runs):
            # initalize methods
            query_strategy = models.AdaptiveQueryStrategy(conf)
            
            # initial unlabeled soundscapes
            all_soundscape_basenames = ['soundscape_{}'.format(idx) for idx in range(conf.n_soundscapes)]
            remaining_soundscape_basenames = all_soundscape_basenames
            remaining_soundscape_basenames = sorted(remaining_soundscape_basenames)

            bns = remaining_soundscape_basenames
        
            budget_count = 0
            # print("####################################################")
            # print("Run: {}".format(idx_run))
            # print("####################################################")

            while budget_count < conf.n_soundscapes:
                sys.stdout.write("Class name: {}, run: {}/{}, annotated soundscapes: {}/{}   \r".format(conf.class_name, idx_run+1, conf.n_runs, budget_count, conf.n_soundscapes))
                sys.stdout.flush()
                #print("----------------- {} -----------------------".format(budget_count))

                f1_train, miou_train, _, _, bns, soundscape_preds, used_queries = simulate_strategy(
                    query_strategy       = query_strategy,
                    soundscape_basenames = bns,
                    n_queries            = conf.n_queries,
                    base_dir             = conf.train_base_dir,
                    min_iou              = conf.min_iou,
                    noise_factor         = conf.noise_factor,
                    normalize            = conf.normalize_embeddings,
                    iteration            = budget_count,
                    emb_win_length       = conf.emb_win_length,
                    fp_noise             = conf.fp_noise,
                    fn_noise             = conf.fn_noise,
                    prominence_threshold = conf.prominence_threshold,
                    converage_threshold  = conf.coverage_threshold,
                )

                query_count[idx_run] += used_queries

                annotated_soundscape_basename, annotations = soundscape_preds
                train_annotation_dir = os.path.join(conf.sim_dir, str(idx_run), 'train_annotations')
                if not os.path.exists(train_annotation_dir):
                    os.makedirs(train_annotation_dir)

                # save annotations in .tsv format
                with open(os.path.join(train_annotation_dir, "iter_{}_".format(budget_count) + annotated_soundscape_basename + '.tsv'), 'w') as f:
                    f.write('onset\toffset\tevent_label\n')
                    for (onset, offset) in annotations:
                        f.write('{}\t{}\t{}\n'.format(onset, offset, conf.class_name))

                f1_scores_train_online[idx_run, budget_count]   = f1_train
                miou_scores_train_online[idx_run, budget_count] = miou_train

                # increase budget count
                budget_count += 1

        conf.save_config()
        #print("done! saving results in {} ...".format(os.path.join(args.results_dir, args.class_name)))

        # shape (n_query_strategies, conf.n_runs, 1, n_soundscapes_budget)
        np.save(os.path.join(conf.sim_dir, "f1_scores_train_online.npy"), f1_scores_train_online)
        np.save(os.path.join(conf.sim_dir, "miou_scores_train_online.npy"), miou_scores_train_online)
        np.save(os.path.join(conf.sim_dir, "query_count.npy"), query_count)
        print("query count: ", query_count)

    # TODO: testing
    #conf.load_config(conf.sim_dir)
    #print("config: {}".format(conf.__dict__))

    #######################################################################################
    # Make predictions using evaluation model trained on the training dataset
    #######################################################################################
    # TODO: check all runs, and only start from the ones that are missing ...
    if not os.path.exists(os.path.join(conf.sim_dir, '0', 'test_scores', conf.model_name)):
        print("Predicting test and train data ...")
        evaluate.predict_test_and_train(conf)

    #######################################################################################
    # Evaluate the predictions
    #######################################################################################
    sound_event_eval.evaluate_test_and_train(conf)

if __name__ == '__main__':
    conf = config.Config()
    run(conf)
