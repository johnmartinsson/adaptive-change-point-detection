import os
import yaml

class Config():
    def __init__(self):
        # Query strategy hyperparameters
        self.prominence_threshold = 0.0

        # Oracle hyperparamters
        self.coverage_threshold = 0.05

        # Simulation hyperparameters
        self.n_soundscapes        = 300
        self.n_queries            = 7
        self.n_runs               = 1
        self.emb_win_length       = 1.0
        self.fp_noise             = 0.0
        self.fn_noise             = 0.0
        self.strategy_name        = 'CPD'
        self.normal_prototypes    = True
        self.normalize_embeddings = True
        self.noise_factor         = 0.0
        self.min_iou              = 0.000001

        # Data hyperparameters
        self.class_name           = 'baby'
        self.snr                  = 0.0
        self.results_dir          = 'results/default'
        self.base_dir             = '/mnt/storage_1/john/data/bioacoustic_sed/'

        # Prediction hyperparameters
        self.evaluation_budgets = [1.0]
        self.model_name         = 'prototypical'

        # Evaluation hyperparameters
        self.t_collar = 2.0

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