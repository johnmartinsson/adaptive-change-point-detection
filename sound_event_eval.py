from __future__ import print_function, absolute_import
import os
import sed_eval
import dcase_util
import numpy as np
import glob
import argparse

import sys
sys.path.append('../')
import metrics

def load_prediction_files(sim_dir, method_name, model_name, idx_run, budget_name):
    """Load the prediction files for a given method and run

    Parameters
    ----------
    sim_dir : str
        Path to the directory containing the simulations.
    method_name : str
        The name of the method to load the predictions for.
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
    train_est_list = glob.glob(os.path.join(sim_dir, method_name, str(idx_run), 'train_scores', 'event_based', '*.txt'))

    test_pattern = os.path.join(sim_dir, method_name, str(idx_run), 'test_scores', model_name, budget_name, 'event_based', '*.txt')
    test_est_list  = glob.glob(test_pattern)

    #print("test_pattern: ", test_pattern)
    #print("train_est_list: ", train_est_list)
    #print("test_est_list: ", test_est_list)

    return sorted(train_est_list), sorted(test_est_list)

def load_reference_files(reference_dir):
    ref_files = glob.glob(os.path.join(reference_dir, "*.txt"))
    ref_files = [f for f in ref_files if 'birdnet' not in f]

    return sorted(ref_files)

def load_file_pair_lists(data_dir, sim_dir, method_name, model_name, idx_run, budget_name):
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

    train_est_list, test_est_list = load_prediction_files(sim_dir, method_name, model_name, idx_run, budget_name)

    train_ref_list = load_reference_files(os.path.join(data_dir, 'train_soundscapes_snr_0.0'))
    test_ref_list  = load_reference_files(os.path.join(data_dir, 'test_soundscapes_snr_0.0'))

    # TODO: make train_scores part of main script, and assert this here
    #assert len(train_est_list) == len(train_ref_list), "Number of training predictions and references does not match! [{:d}] [{:d}]".format(len(train_est_list), len(train_ref_list))
    assert len(test_est_list) == len(test_ref_list), "Number of test predictions and references does not match! [{:d}] [{:d}]".format(len(test_est_list), len(test_ref_list))

    return zip(train_ref_list, train_est_list), zip(test_ref_list, test_est_list)

def evaluate(file_pair_list, t_collar=0.200):
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
    segment_based_metrics = sed_eval.sound_event.SegmentBasedMetrics(event_labels)
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

    f1_segment_based = segment_based_metrics.results_overall_metrics()['f_measure']['f_measure']
    f1_event_based   = event_based_metrics.results_overall_metrics()['f_measure']['f_measure']

    # TODO: maybe add the intersection-over-union metric as well?
    
    return f1_event_based, f1_segment_based

def main():
    """Main
    """
    parser = argparse.ArgumentParser(description='Evaluate the sound event detection performance of a method.')
    parser.add_argument('--class_name', type=str, help='The name of the method to evaluate.')
    parser.add_argument('--model_name', type=str, help='The name of the model to evaluate.')
    #parser.add_argument('--budget_name', type=str, help='The name of the budget to evaluate.')
    parser.add_argument('--sim_dir', type=str, help='The path to the directory containing the simulation results.')
    parser.add_argument('--t_collar', type=float, help='The tolerance collar to use for the evaluation.')
    parser.add_argument('--n_runs', type=int, help='The number of runs to evaluate.')
    parser.add_argument('--base_dir', type=str, help='The base directory.')
    parser.add_argument('--only_budget_1', required=True, type=str)
    args = parser.parse_args()

    n_runs      = args.n_runs
    model_name  = args.model_name
    #budget_name = args.budget_name
    class_name  = args.class_name
    sim_dir     = args.sim_dir
    t_collar    = args.t_collar

    data_dir = '{}/generated_datasets/{}_{:.1f}_{:.2f}s/'.format(args.base_dir, class_name, 1.0, 0.25)
        
    if not os.path.exists(data_dir):
        raise IOError("Directory does not exist [{:s}]".format(data_dir))
    
    if not os.path.exists(sim_dir):
        raise IOError("Directory does not exist [{:s}]".format(sim_dir))

    if args.only_budget_1 == 'True':
        budget_names = ['budget_1.0']
    else:
        budget_names = ['budget_0.01', 'budget_0.02', 'budget_0.04', 'budget_0.08', 'budget_0.16', 'budget_0.32', 'budget_0.64', 'budget_1.0']

    n_budgets = len(budget_names)

    f1_event_based_train_results   = np.zeros((4, n_runs, n_budgets))
    f1_segment_based_train_results = np.zeros((4, n_runs, n_budgets))
    f1_event_based_test_results    = np.zeros((4, n_runs, n_budgets))
    f1_segment_based_test_results  = np.zeros((4, n_runs, n_budgets))

    for idx_budget, budget_name in enumerate(budget_names):
        for idx_method, method_name in enumerate(['OPT', 'ADP', 'CPD', 'FIX']):
            for idx_run in range(n_runs):
                sys.stdout.write("\rEvaluating method {:s} run {:d} budget {:s}".format(method_name, idx_run, budget_name))
                sys.stdout.flush()

                # create the file list
                train_file_list, test_file_list = load_file_pair_lists(data_dir, sim_dir, method_name, model_name, idx_run, budget_name)

                # evaluate the training set label quality
                # TODO: I need to re-think this evaluation
                f1_event_based_train, f1_segment_based_train = evaluate(train_file_list, t_collar=t_collar)

                f1_event_based_train_results[idx_method, idx_run, idx_budget]   = f1_event_based_train
                f1_segment_based_train_results[idx_method, idx_run, idx_budget] = f1_segment_based_train

                # evaluate the test set prediction quality
                f1_event_based_test, f1_segment_based_test = evaluate(test_file_list, t_collar=t_collar)

                f1_event_based_test_results[idx_method, idx_run, idx_budget]   = f1_event_based_test
                f1_segment_based_test_results[idx_method, idx_run, idx_budget] = f1_segment_based_test

    np.save(os.path.join(sim_dir, '{}_f1_event_based_train_results.npy'.format(model_name)), f1_event_based_train_results)
    np.save(os.path.join(sim_dir, '{}_f1_segment_based_train_results.npy'.format(model_name)), f1_segment_based_train_results)
    np.save(os.path.join(sim_dir, '{}_f1_event_based_test_results.npy'.format(model_name)), f1_event_based_test_results)
    np.save(os.path.join(sim_dir, '{}_f1_segment_based_test_results.npy'.format(model_name)), f1_segment_based_test_results)

    # print("##############################################")
    # print("Class: ", class_name)
    # print("##############################################")
    # print("")
    # print("Train labels")
    # print("")

    # print("Method & F1 (event-based) & F1 (segment-based) \\\\")
    # print("\\hline")
    # for idx_method, method_name in enumerate(['OPT', 'ADP', 'CPD', 'FIX']):
    #     # print method name and f1 {mean} +/- {std} formatted for a LaTeX table
    #     print("{:s} & {:.3f} $\pm$ {:.3f} & {:.3f} $\pm$ {:.3f} \\\\".format(method_name, np.mean(f1_event_based_train_results[idx_method]), np.std(f1_event_based_train_results[idx_method]), np.mean(f1_segment_based_train_results[idx_method]), np.std(f1_segment_based_train_results[idx_method])))
    
    # print("")
    # print("Test predictions")
    # print("")
    # print("Method & F1 (event-based) & F1 (segment-based) \\\\")
    # print("\\hline")
    # for idx_method, method_name in enumerate(['OPT', 'ADP', 'CPD', 'FIX']):
    #     # print method name and f1 {mean} +/- {std} formatted for a LaTeX table
    #     print("{:s} & {:.3f} $\pm$ {:.3f} & {:.3f} $\pm$ {:.3f} \\\\".format(method_name, np.mean(f1_event_based_test_results[idx_method]), np.std(f1_event_based_test_results[idx_method]), np.mean(f1_segment_based_test_results[idx_method]), np.std(f1_segment_based_test_results[idx_method])))

    # print("")
    # print("")
if __name__ == "__main__":
    main()