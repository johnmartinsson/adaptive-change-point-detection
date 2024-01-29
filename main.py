import run_simulation
import config

class SearchSpace(object):
    def __init__(self):
        pass

    def __str__(self):
        pass


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


    conf = config.Config()

    print('###################################################################')
    print("# Running the main experiment ....")
    print('###################################################################')
    # TODO: parallelize this!
    for strategy_name in strategy_names:
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
                        conf.n_runs               = n_runs
                        conf.results_dir          = results_dir

                        # run the simulation
                        print("config: {}".format(conf.__dict__))
                        run_simulation.run(conf)

if __name__ == '__main__':
    main()