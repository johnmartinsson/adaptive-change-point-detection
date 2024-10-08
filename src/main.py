import os
import run_simulation
import config
import concurrent.futures

class SearchSpace(object):
    def __init__(self):
        pass

    def __str__(self):
        pass

def run_strategy(strategy_name): #, n_queriess, prominence_thresholds, coverage_thresholds, class_names, model_names, n_runs, results_dir):
    conf = config.Config()

    # TODO: make a search space class
    # search space
    n_queriess            = [3,5,7,9,11,13]
    prominence_thresholds = [0.0]
    coverage_thresholds   = [0.5]
    class_names           = ['me', 'dog', 'baby']
    model_names           = ['prototypical', 'mlp']
    n_runs                = 2
    results_dir           = './results/eusipco_2024_reproduced'
    noises                = [0.0, 0.2]
    
    conf.results_dir = results_dir
    conf.n_runs      = n_runs
    
    for n_queries in n_queriess:
        for prominence_threshold in prominence_thresholds:
            for coverage_threshold in coverage_thresholds:
                for class_name in class_names:
                    for model_name in model_names:
                        for noise in noises:
                            # set the config parameters
                            conf.fn_noise             = noise
                            conf.fp_noise             = noise
                            conf.strategy_name        = strategy_name
                            conf.n_queries            = n_queries
                            conf.prominence_threshold = prominence_threshold
                            conf.coverage_threshold   = coverage_threshold
                            conf.class_name           = class_name
                            conf.model_name           = model_name

                            # run the simulation
                            print("Running simulation with the following configuration:")
                            conf.pretty_print()
                            run_simulation.run(conf)

def main():

    print('###################################################################')
    print("# Running the main experiment ....")
    print('###################################################################')
    # parallellize over startegy names
    strategy_names        = ['OPT', 'ADP', 'CPD', 'FIX']

    # TODO: create all configs here and parallelize over them instead of methods.
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(run_strategy, strategy_names)

if __name__ == '__main__':
    main()
