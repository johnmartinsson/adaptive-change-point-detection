from ray import tune
from ray import air
from ray.tune import CLIReporter

import argparse

from functools import partial

import utils
import search_spaces

def run_experiment(config, data_dir):

    # active learning
    active_dataset = utils.get_active_dataset_by_config(config, data_dir)
    active_learner = utils.get_active_learner_by_config(config)
    oracle         = utils.get_oracle_by_config(config)

    # evaluation
    testdataset    = utils.get_test_dataset_by_config(config, data_dir)

    n_queries = config['labeling_budget'] // config['batch_size']

    print("n_queries: ", n_queries)
    
    scores = []
    for q_idx in range(n_queries):
        print("--------------------------------------")
        print("Query index = {}".format(q_idx))
        print("--------------------------------------")
        query_timings   = active_learner.predict_query_timings(active_dataset)
        labels, timings = oracle.annotate(query_timings)

        # update active dataset
        active_dataset.update(labels, timings)

        # update active learner
        active_learner.update(active_dataset)

        # evaluate
        score = active_learner.evaluate(testdataset)
        scores.append(score)

def main():
    parser = argparse.ArgumentParser(description='Hyperparameter search.')
    parser.add_argument('--labeling_budget', help='The labeling budget (iterations * batch_size).', required=True, type=int)
    parser.add_argument('--num_samples', help='The number of samples from the search space.', required=True, type=int)
    parser.add_argument('--name', help='The name of the hyperparamter search experiment.', required=True, type=str)
    parser.add_argument('--ray_root_dir', help='The name of the directory to save the ray search results.', required=True, type=str)
    parser.add_argument('--data_dir', help='The absolute path to the soundscape directory.', required=True, type=str)
    args = parser.parse_args()

    if "development" in args.name:
        search_space = search_spaces.development(args.labeling_budget)

    # results terminal reporter
    reporter = CLIReporter(
        metric_columns=[
            "iteration",
            "score",
        ],
        parameter_columns = [
            'batch_size',
        ],
        max_column_length = 10
    )

    run_experiment_fn = partial(run_experiment, data_dir=args.data_dir)

    trainable_with_resources = tune.with_resources(run_experiment_fn, {"cpu" : 8.0, "gpu": 1.00})

    tuner = tune.Tuner(
        trainable_with_resources,
        param_space = search_space,
        run_config  = air.RunConfig(
            verbose=1,
            progress_reporter = reporter,
            name              = args.name,
            local_dir         = args.ray_root_dir, 
        ),
	tune_config = tune.TuneConfig(
	    num_samples = args.num_samples,
	),
    )

    result = tuner.fit()

    return

if __name__ == '__main__':
    main()
