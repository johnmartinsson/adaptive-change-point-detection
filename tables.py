import os
import glob
import numpy as np
import pandas as pd
import yaml

import config

def main():
    conf = config.Config()
    results_dir = conf.results_dir

    # find all config.yaml files in the results_dir
    config_files = glob.glob(os.path.join(results_dir, '**', 'config.yaml'), recursive=True)
    print(len(config_files))

    conf = config.Config()
    conf.load_config_yaml(config_files[0])


    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    noise_styles = ['-', '--']

    #split = 'Train'
    metric_names = ['F-score', 'Precision', 'Recall']

    model_name = 'mlp'
    train = False

    for train in [True, False]:
        if train:
            model_names = ['prototypical']
        else:
            model_names = ['prototypical', 'mlp']

        for model_name in model_names:
            if train:
                print("##################################################################")
                print("# Table 1")
                print("##################################################################")
            else:
                if model_name == 'prototypical':
                    print("##################################################################")
                    print("# Table 2")
                    print("##################################################################")
                elif model_name == 'mlp':
                    print("##################################################################")
                    print("# Table 3")
                    print("##################################################################")
                else:
                    raise ValueError("Unknown model_name: {}".format(model_name))
            if train:
                event_based_df, segment_based_df = conf.load_train_results(model_name=model_name)
            else:
                event_based_df, segment_based_df = conf.load_test_results(model_name=model_name)
            prominence_threshold = 0.0
            coverage_threshold   = 0.5
            n_queries = 7

            df = segment_based_df[segment_based_df['model_name'] == model_name]
            df = df[df['prominence_threshold'] == prominence_threshold]
            df = df[df['coverage_threshold'] == coverage_threshold]
            df = df[df['n_queries'] == n_queries]

            df_event = event_based_df[event_based_df['model_name'] == model_name]
            df_event = df_event[df_event['prominence_threshold'] == prominence_threshold]
            df_event = df_event[df_event['coverage_threshold'] == coverage_threshold]
            df_event = df_event[df_event['n_queries'] == n_queries]

            new_strategy_names = ['ORC', 'A-CPD', 'F-CPD', 'FIX']

            for idx_metric, metric_name in enumerate(['f_measure']):
                for idx_noise, noise in enumerate([0.0]): #, 0.2]):
                    segment_stds = []
                    event_stds = []
                    if train:
                        print("Strategy & \multicolumn{2}{|c|}{Meerkat} & \multicolumn{2}{|c|}{Dog} & \multicolumn{2}{|c|}{Baby} \\\\")
                        print("         & Segment & Event               & Segment & Event           & Segment & Event \\\\")
                    else:
                        print("Strategy & Meerkat & Dog & Baby \\\\")
                    print("\hline")
                    for idx_strat, strategy_name in enumerate(['OPT', 'ADP', 'CPD', 'FIX']):

                        row_str = new_strategy_names[idx_strat]
                        for class_name in ['me', 'dog', 'baby']:    
                            _df = df[df['strategy_name'] == strategy_name]
                            _df = _df[_df['fn_noise'] == noise]
                            _df = _df[_df['fp_noise'] == noise]
                            _df = _df[_df['class_name'] == class_name]
                            average_df = _df.groupby(['n_queries'])[metric_name].mean().reset_index()
                            std_df = _df.groupby(['n_queries'])[metric_name].std().reset_index()

                            segment_stds.append(std_df[metric_name].values[0])

                            n_runs = 10
                            n_classes = 3
                            assert(len(_df) == n_runs)

                            _df_event = df_event[df_event['strategy_name'] == strategy_name]
                            _df_event = _df_event[_df_event['fn_noise'] == noise]
                            _df_event = _df_event[_df_event['fp_noise'] == noise]
                            _df_event = _df_event[_df_event['class_name'] == class_name]

                            assert(len(_df) == n_runs)

                            average_df_event = _df_event.groupby(['n_queries'])[metric_name].mean().reset_index()
                            std_df_event = _df_event.groupby(['n_queries'])[metric_name].std().reset_index()

                            event_stds.append(std_df_event[metric_name].values[0])
                            
                            if train:
                                row_str += " & ${:.2f}$ & ${:.2f}$".format(average_df[metric_name].values[0], average_df_event[metric_name].values[0])
                            else:
                                row_str += " & ${:.2f}$".format(average_df[metric_name].values[0])

                        row_str += " \\\\"
                        print(row_str)
                        if strategy_name == 'OPT':
                            print("\\hline")

            print("")
            print("")
            print("")
        

if __name__ == "__main__":
    main()