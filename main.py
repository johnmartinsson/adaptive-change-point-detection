import run_simulation
import config
import concurrent.futures
import functools

class SearchSpace(object):
    def __init__(self):
        pass

    def __str__(self):
        pass

def run_strategy(strategy_name, n_queriess, prominence_thresholds, coverage_thresholds, class_names, n_runs, results_dir):
    conf = config.Config()
    
    conf.results_dir = results_dir
    conf.n_runs      = n_runs

    for n_queries in n_queriess:
        for prominence_threshold in prominence_thresholds:
            for coverage_threshold in coverage_thresholds:
                for class_name in class_names:
                    
                    # set the config parameters
                    conf.strategy_name        = strategy_name
                    conf.n_queries            = n_queries
                    conf.prominence_threshold = prominence_threshold
                    conf.coverage_threshold   = coverage_threshold
                    conf.class_name           = class_name

                    # run the simulation
                    print("config: {}".format(conf.__dict__))
                    run_simulation.run(conf)

def main():

    # TODO: make a search space class
    # search space
    strategy_names        = ['OPT', 'ADP', 'CPD', 'FIX']
    n_queriess            = [7, 9, 11, 13, 15, 17, 19]
    prominence_thresholds = [0.0, 0.1]
    coverage_thresholds   = [0.05, 0.95]
    class_names           = ['me', 'dog', 'baby']
    n_runs                = 3
    results_dir           = 'results/2024-01-29'

    print('###################################################################')
    print("# Running the main experiment ....")
    print('###################################################################')
    # TODO: parallelize this!

    # parallellize over startegy names

    run_strategy_fn = functools.partial(
        run_strategy,
        n_queriess            = n_queriess,
        prominence_thresholds = prominence_thresholds,
        coverage_thresholds   = coverage_thresholds,
        class_names           = class_names,
        n_runs                = n_runs,
        results_dir           = results_dir
    )

    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(run_strategy_fn, strategy_names)

if __name__ == '__main__':
    main()