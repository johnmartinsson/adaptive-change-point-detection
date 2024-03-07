from __future__ import print_function, absolute_import
import os
import sed_eval
import dcase_util
import numpy as np
import glob
import argparse
import yaml

import sys
sys.path.append('../')
import metrics

def load_prediction_files(sim_dir, model_name, idx_run, budget_name):
    """Load the prediction files for a given method and run

    Parameters
    ----------
    sim_dir : str
        Path to the directory containing the simulations.
    model_name : str
        The name of the model to load the predictions for.
    idx_run : int
        The index of the run to load the predictions for.

    Returns
    -------
    list
        List containing the estimated event list filenames.
    """

    # TODO: how did I compute the train_scores?
    train_est_list = glob.glob(os.path.join(sim_dir, str(idx_run), 'train_scores', 'event_based', '*.txt'))

    test_pattern = os.path.join(sim_dir, str(idx_run), 'test_scores', model_name, budget_name, 'event_based', '*.txt')
    test_est_list  = glob.glob(test_pattern)

    #print("test_pattern: ", test_pattern)
    #print("train_est_list: ", train_est_list)
    #print("test_est_list: ", test_est_list)

    return sorted(train_est_list), sorted(test_est_list)

def load_reference_files(reference_dir):
    ref_files = glob.glob(os.path.join(reference_dir, "*.txt"))
    ref_files = [f for f in ref_files if 'birdnet' not in f]

    return sorted(ref_files)

def load_file_pair_lists(train_base_dir, test_base_dir, sim_dir, strategy_name, model_name, idx_run, budget_name):
    """Create file list for evaluation

    Parameters
    ----------
    sim_dir : str
        Path to the directory containing the predictions.

    Returns
    -------
    (list, list)
        Lists of tuples containing the reference and estimated event list filenames for the training and test predictions.
    """

    def basename(f):
        return os.path.splitext(os.path.basename(f))[0]

    train_est_list, test_est_list = load_prediction_files(sim_dir, model_name, idx_run, budget_name)

    train_ref_list = load_reference_files(train_base_dir)
    test_ref_list  = load_reference_files(test_base_dir)

    # TODO: make train_scores part of main script, and assert this here
    #assert len(train_est_list) == len(train_ref_list), "Number of training predictions and references does not match! [{:d}] [{:d}]".format(len(train_est_list), len(train_ref_list))
    assert len(test_est_list) == len(test_ref_list), "Number of test predictions and references does not match! [{:d}] [{:d}]".format(len(test_est_list), len(test_ref_list))

    return zip(train_ref_list, train_est_list), zip(test_ref_list, test_est_list)

def evaluate(file_pair_list, t_collar=0.200, time_resolution=0.01):
    data = []
    all_data = dcase_util.containers.MetaDataContainer()
    for ref_file, est_file in file_pair_list:
        assert os.path.basename(ref_file) == os.path.basename(est_file), "Reference and estimated file names do not match! [{:s}] [{:s}]".format(ref_file, est_file)
        reference_event_list = sed_eval.io.load_event_list(
            os.path.abspath(ref_file)
        )

        estimated_event_list = sed_eval.io.load_event_list(
            os.path.abspath(est_file)
        )

        data.append({
            'reference_event_list': reference_event_list,
            'estimated_event_list': estimated_event_list
        })

        all_data += reference_event_list

    event_labels = all_data.unique_event_labels

    # TODO: make sure sensible settings are used
    #print("time_resolution: ", time_resolution)
    segment_based_metrics = sed_eval.sound_event.SegmentBasedMetrics(event_labels, time_resolution=time_resolution)
    #print("t_collar: ", t_collar)
    event_based_metrics   = sed_eval.sound_event.EventBasedMetrics(event_labels, t_collar=t_collar)

    # TODO: understand how results are accumulated, is it an online average over all files?
    #miou_files = []
    
    for file_pair in data:
        segment_based_metrics.evaluate(
            file_pair['reference_event_list'],
            file_pair['estimated_event_list']
        )

        event_based_metrics.evaluate(
            file_pair['reference_event_list'],
            file_pair['estimated_event_list']
        )

        # compute the class average intersection-over-union
    #    miou_file = metrics.class_average_miou(
    #        events_ref  = file_pair['reference_event_list'],
    #        events_pred = file_pair['estimated_event_list'],
    #        n_classes   = len(event_labels),
    #    )
        
    #    miou_files.append(miou_file)
    #iou_event_based  = np.mean(miou_files)

    segment_based = segment_based_metrics.results_overall_metrics()#['f_measure']['f_measure']
    event_based   = event_based_metrics.results_overall_metrics()#['f_measure']['f_measure']

    # TODO: maybe add the intersection-over-union metric as well?
    
    return event_based, segment_based

def evaluate_test_and_train(conf):
    """Evaluate the test and train set predictions for all methods and runs"""
    #print("evaluation ...")
        
    if not os.path.exists(conf.train_base_dir):
        print("Directory does not exist [{:s}]".format(conf.train_base_dir))
        raise IOError("Directory does not exist [{:s}]".format(conf.train_base_dir))
    
    if not os.path.exists(conf.test_base_dir):
        print("Directory does not exist [{:s}]".format(conf.test_base_dir))
        raise IOError("Directory does not exist [{:s}]".format(conf.test_base_dir))
    
    if not os.path.exists(conf.sim_dir):
        print("Directory does not exist [{:s}]".format(conf.sim_dir))
        raise IOError("Directory does not exist [{:s}]".format(conf.sim_dir))


    budget_names = conf.budget_names
    #print("budget names: ", budget_names)

    #n_budgets = len(budget_names)

    for idx_budget, budget_name in enumerate(budget_names):
        #print("budget name ...", budget_name)
        for idx_run in range(conf.n_runs):
            #print("idx_run ...", idx_run)
            sys.stdout.write("\rEvaluating method {:s} run {:d} budget {:s}\n".format(conf.strategy_name, idx_run, budget_name))
            sys.stdout.flush()

            # create the file list
            train_file_list, test_file_list = load_file_pair_lists(conf.train_base_dir, conf.test_base_dir, conf.sim_dir, conf.strategy_name, conf.model_name, idx_run, budget_name)

            run_dir = os.path.join(conf.sim_dir, str(idx_run))
            #print("run_dir: ", run_dir)

            # evaluate the training set label quality
            # TODO: I need to re-think this evaluation
            event_based_train, segment_based_train = evaluate(train_file_list, t_collar=conf.t_collar, time_resolution=conf.time_resolution)

            #print("event_based_train: ", event_based_train)
            # save the dictionary to a file
            with open(os.path.join(run_dir, 'event_based_train_metrics.yaml'), 'w') as f:
                yaml.dump(event_based_train, f)
            
            with open(os.path.join(run_dir, 'segment_based_train_metrics.yaml'), 'w') as f:
                yaml.dump(segment_based_train, f)

            # evaluate the test set prediction quality
            event_based_test, segment_based_test = evaluate(test_file_list, t_collar=conf.t_collar, time_resolution=conf.time_resolution)

            # save the dictionary to a file
            # run_dir/test_scores/prototypical/budget_1.0/'
            with open(os.path.join(run_dir, 'test_scores', conf.model_name, budget_name, 'event_based_test_metrics.yaml'), 'w') as f:
                yaml.dump(event_based_test, f)
            
            with open(os.path.join(run_dir, 'test_scores', conf.model_name, budget_name, 'segment_based_test_metrics.yaml'), 'w') as f:
                yaml.dump(segment_based_test, f)

# if __name__ == "__main__":
#     main()