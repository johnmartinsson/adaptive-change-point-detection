import os
import yaml
import pandas as pd
import glob

class Config():
    def __init__(self, sim_dir=None):
        # Query strategy hyperparameters
        self.prominence_threshold = 0.0

        # Oracle hyperparamters
        self.coverage_threshold = 0.5

        # Simulation hyperparameters
        self.n_soundscapes        = 300
        self.n_queries            = 7
        self.n_runs               = 1
        self.emb_win_length       = 1.0
        self.fp_noise             = 0.0
        self.fn_noise             = 0.0
        self.strategy_name        = 'ADP'
        self.normal_prototypes    = True
        self.normalize_embeddings = True
        self.noise_factor         = 0.0
        self.min_iou              = 0.000001

        # Data hyperparameters
        self.class_name           = 'baby'
        self.snr                  = 0.0
        self.results_dir          = '/mnt/storage_1/john/al_for_sed_results/2024-03-12/'   #'./eusipco_2024'
        #self.base_dir             = '/mnt/storage_1/john/data/bioacoustic_sed/'
        self.base_dir             = '/mnt/storage_1/john/data/bioacoustic_sed_2024_02_22/' #'./data/generated_datasets' 

        # Prediction hyperparameters
        self.evaluation_budgets = [1.0]
        self.model_name         = 'prototypical'

        # Evaluation hyperparameters
        self.t_collar           = 0.5
        self.time_resolution    = 0.05

        # load the config from the given sim_dir
        if sim_dir is not None:
            self.load_config(sim_dir)

    def pretty_print(self):
        """ Print the config in a nice format """

        # align the values to the right
        max_len = max([len(attr) for attr in vars(self)])
        print("-" * (max_len + 1 + 20))
        print("Config:")
        print("-" * (max_len + 1 + 20))
        for attr in vars(self):
            print("\t{:>{width}}: {}".format(attr, getattr(self, attr), width=max_len))
        print("-" * (max_len + 1 + 20))
        print("")

    def save_config(self):
        """ Save the config properties to a YAML file """
        config_dict = {attr: getattr(self, attr) for attr in vars(self)}
        with open(os.path.join(self.sim_dir, 'config.yaml'), 'w') as f:
            print("Saving config to {}".format(os.path.join(self.sim_dir, 'config.yaml')))
            yaml.dump(config_dict, f)

    def load_config(self, sim_dir):
        # TODO: implement the load config function
        with open(os.path.join(sim_dir, 'config.yaml'), 'r') as f:
            config_dict = yaml.safe_load(f)

            # set all attribute in self to the corresponding value in config_dict
            for attr in config_dict:
                setattr(self, attr, config_dict[attr])
    
    def load_config_yaml(self, yaml_file):
        with open(yaml_file, 'r') as f:
            config_dict = yaml.safe_load(f)

            # set all attribute in self to the corresponding value in config_dict
            for attr in config_dict:
                setattr(self, attr, config_dict[attr])
    
    def load_test_results(self, model_name='prototypical'):
        """ Load all the results from the results directory """

        segment_based_dicts = []
        event_based_dicts   = []

        config_files = glob.glob(os.path.join(self.results_dir, '**', 'config.yaml'), recursive=True)

        for config_file in config_files:
            conf = Config()
            conf.load_config_yaml(config_file)
            conf.model_name = model_name
            
            for idx_run in range(conf.n_runs):
                for budget in conf.evaluation_budgets:
                    budget_name = 'budget_{}'.format(budget)

                    run_dir = os.path.join(conf.sim_dir, str(idx_run))

                    # segment based
                    segment_based_test_metrics_file = os.path.join(run_dir, 'test_scores', conf.model_name, budget_name, 'segment_based_test_metrics.yaml')
                    if os.path.exists(segment_based_test_metrics_file):
                        with open(segment_based_test_metrics_file, 'r') as f:
                            segment_based_test = yaml.safe_load(f)
                        
                        setting_values = {attr: getattr(conf, attr) for attr in vars(conf)}
                        
                        result_names   = ['f_measure', 'precision', 'recall']

                        result_values  = {result_name: segment_based_test['f_measure'][result_name] for result_name in result_names}
                        run_and_budget = {'run': idx_run, 'budget': budget}
                        dict_to_append = {**setting_values, **result_values, **run_and_budget}
                        segment_based_dicts.append(dict_to_append)
                    else:
                        print("File not found: {}".format(segment_based_test_metrics_file))

                    # event based
                    event_based_test_metrics_file = os.path.join(run_dir, 'test_scores', conf.model_name, budget_name, 'event_based_test_metrics.yaml')
                    if os.path.exists(event_based_test_metrics_file):
                        with open(event_based_test_metrics_file, 'r') as f:
                            event_based_test = yaml.safe_load(f)

                        result_values  = {result_name: event_based_test['f_measure'][result_name] for result_name in result_names}
                        dict_to_append = {**setting_values, **result_values, **run_and_budget}
                        event_based_dicts.append(dict_to_append)
                    else:
                        print("File not found: {}".format(event_based_test_metrics_file))

        segment_based_df = pd.DataFrame(segment_based_dicts)
        event_based_df   = pd.DataFrame(event_based_dicts)

        return event_based_df, segment_based_df

    
    def load_train_results(self, model_name='prototypical'):
        """ Load all the results from the results directory """

        segment_based_dicts = []
        event_based_dicts   = []

        config_files = glob.glob(os.path.join(self.results_dir, '**', 'config.yaml'), recursive=True)

        for config_file in config_files:
            conf = Config()
            conf.load_config_yaml(config_file)
            conf.model_name = model_name
            
            for idx_run in range(conf.n_runs):
                run_dir = os.path.join(conf.sim_dir, str(idx_run))

                # segment based
                if os.path.exists(os.path.join(run_dir, 'segment_based_train_metrics.yaml')):
                    with open(os.path.join(run_dir, 'segment_based_train_metrics.yaml'), 'r') as f:
                        segment_based_test = yaml.safe_load(f)
                    
                    setting_values = {attr: getattr(conf, attr) for attr in vars(conf)}
                    
                    result_names   = ['f_measure', 'precision', 'recall']

                    result_values  = {result_name: segment_based_test['f_measure'][result_name] for result_name in result_names}
                    run_and_budget = {'run': idx_run}
                    dict_to_append = {**setting_values, **result_values, **run_and_budget}
                    segment_based_dicts.append(dict_to_append)
                else:
                    print("File not found: {}".format(os.path.join(run_dir, 'segment_based_train_metrics.yaml')))

                # event based
                if os.path.exists(os.path.join(run_dir, 'event_based_train_metrics.yaml')):
                    with open(os.path.join(run_dir, 'event_based_train_metrics.yaml'), 'r') as f:
                        event_based_test = yaml.safe_load(f)

                    result_values  = {result_name: event_based_test['f_measure'][result_name] for result_name in result_names}
                    dict_to_append = {**setting_values, **result_values, **run_and_budget}
                    event_based_dicts.append(dict_to_append)
                else:
                    print("File not found: {}".format(os.path.join(run_dir, 'event_based_train_metrics.yaml')))

        segment_based_df = pd.DataFrame(segment_based_dicts)
        event_based_df   = pd.DataFrame(event_based_dicts)

        return event_based_df, segment_based_df

    @property
    def budget_names(self):
        return ['budget_{}'.format(b) for b in self.evaluation_budgets]

    @property
    def sim_dir(self):
        return os.path.join(self.results_dir, self.class_name, self.settings_string, self.strategy_name)
    
    @property
    def settings_string(self):
        return 'n_queries_{}_noise_{}_prominence_threshold_{}_coverage_threshold_{}'.format(self.n_queries, self.fn_noise, self.prominence_threshold, self.coverage_threshold)


    @property
    def train_base_dir(self):
        return os.path.join(self.base_dir, 'generated_datasets/{}_1.0_0.25s'.format(self.class_name), 'train_soundscapes_snr_{}'.format(self.snr))
    
    @property
    def test_base_dir(self):
        return self.train_base_dir.replace('train', 'test')