import os
import glob
import numpy as np

import config
import datasets
import models
import visualize

import matplotlib.pyplot as plt


def main():
    # A-CPD
    conf = config.Config()
    # the simulation directory with the A-CPD annotations
    sim_dir = conf.sim_dir
    print("sim_dir: ", sim_dir)
    # load the annotated embeddings
    p_embs, n_embs = datasets.load_annotated_embeddings(conf, sim_dir)
    conf.strategy_name = 'ADP'
    adp_model = models.AdaptiveQueryStrategy(conf)
    # update the model with the annotated embeddings
    adp_model.update(p_embs, n_embs)

    # F-CPD
    conf = config.Config()
    conf.strategy_name = 'CPD'
    cpd_model = models.AdaptiveQueryStrategy(conf)

    # FIX
    conf = config.Config()
    conf.strategy_name = 'FIX'
    fix_model = models.AdaptiveQueryStrategy(conf)

    # OPT
    conf = config.Config()
    conf.strategy_name = 'OPT'
    opt_model = models.AdaptiveQueryStrategy(conf)

    test_base_dir = conf.test_base_dir #train_base_dir.replace('train', 'test')

    new_query_strategy_names = ['A-CPD', 'F-CPD', 'FIX']

    fig_dir = 'results/figures_reproduced/'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    soundscape_number = 1

    ###########################################################################
    # Fig. 1
    ###########################################################################
    visualize.visualize_query_strategies(
        [adp_model, cpd_model, fix_model],
        ["ADP", "CPD", "FIX"], #, "FIX"],
        new_query_strategy_names,
        #"FIX, max_n_queries = {}".format(n_queries),
        "soundscape_{}".format(soundscape_number),
        test_base_dir,
        n_queries     = 7,
        savefile      = os.path.join(fig_dir, 'figure_2.png'.format(soundscape_number)),
        noise_factor  = 0,
        normalize     = False,
        coverage_threshold=0.5,
        prominence_threshold=0,
    )

    ###########################################################################
    # Fig. 2
    ###########################################################################
    visualize.visualize_concept(
        [opt_model, fix_model],
        ["Opt.", "Sub."], #, "FIX"],
        #"FIX, max_n_queries = {}".format(n_queries),
        "soundscape_{}".format(soundscape_number),
        test_base_dir,
        n_queries     = 7,
        savefile      = os.path.join(fig_dir, 'figure_1.png'.format(soundscape_number)),
        noise_factor  = 0,
        normalize     = False,
        coverage_threshold=0.5,
        prominence_threshold=0,
    )

    ###########################################################################
    # Fig. 4
    ###########################################################################
    results_dir = conf.results_dir
    # find all config.yaml files in the results_dir
    config_files = glob.glob(os.path.join(results_dir, '**', 'config.yaml'), recursive=True)
    print(len(config_files))
    conf = config.Config()
    conf.load_config_yaml(config_files[0])

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    noise_styles = ['-', '--']

    split = 'Test'
    
    figsize_side_by_side = (5.5, 3)
    n_columns = 2
    n_rows    = 1

    model_names = ['ProtoNet', 'MLP']
    metric_names = ['F-score', 'Precision', 'Recall']

    new_strategy_names = ['ORC', 'A-CPD', 'F-CPD', 'FIX']

    for idx_model, model_name in enumerate(['prototypical', 'mlp']):
        event_based_df, segment_based_df = conf.load_test_results(model_name=model_name)
        prominence_threshold = 0.0
        coverage_threshold   = 0.5

        df = segment_based_df[segment_based_df['model_name'] == model_name]
        df = df[df['prominence_threshold'] == prominence_threshold]
        df = df[df['coverage_threshold'] == coverage_threshold]

        fig, ax = plt.subplots(n_rows, n_columns, figsize=figsize_side_by_side)

        for idx_metric, metric_name in enumerate(['f_measure']):
            values = []
            for idx_noise, noise in enumerate([0.0, 0.2]):
                for idx_strat, strategy_name in enumerate(['OPT', 'ADP', 'CPD', 'FIX']):
                    
                    _df = df[df['strategy_name'] == strategy_name]
                    _df = _df[_df['fn_noise'] == noise]
                    _df = _df[_df['fp_noise'] == noise]
                    average_df = _df.groupby(['n_queries'])[metric_name].mean().reset_index()

                    # plot n_queries vs. f1-score
                    title = r'{} ($\beta = {}$)'.format(model_names[idx_model], noise)

                    ax[idx_noise].set_title(title)

                    ax[idx_noise].set_xlabel(r'Queries per soundscape ($B$)')

                    if idx_noise == 0:
                        ax[idx_noise].set_ylabel(metric_names[idx_metric])

                    if strategy_name == 'OPT':
                        ax[idx_noise].plot(average_df['n_queries'][2:], average_df[metric_name][2:], '-o', color=colors[idx_strat], label='{}'.format(new_strategy_names[idx_strat]))
                    else:
                        ax[idx_noise].plot(average_df['n_queries'], average_df[metric_name], '-o', color=colors[idx_strat], label='{}'.format(new_strategy_names[idx_strat]))

                    ax[idx_noise].set_xticks(average_df['n_queries'], [str(x) for x in average_df['n_queries']])

                    values.append(average_df[metric_name].values)

            values = np.concatenate(values)
            min_value = np.min(values) - 0.03
            max_value = np.max(values) + 0.03
            ax[0].set_ylim(min_value, max_value)
            ax[1].set_ylim(min_value, max_value)

            ax[0].set_xlim(2.5, 13.5)
            ax[1].set_xlim(2.5, 13.5)

            alpha=0.15
            ax[0].fill_between([7, 15], min_value, max_value, alpha=alpha, color='grey')
            ax[1].fill_between([7, 15], min_value, max_value, alpha=alpha, color='grey')

            #plt.ylim(0.1, 0.6)
        plt.tight_layout()
        ax[0].legend()
        plt.savefig(os.path.join(fig_dir, '{}_figure_4.pdf'.format(model_name)), format='pdf', dpi=1200)

    ###########################################################################
    # Fig. 3
    ###########################################################################
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    noise_styles = ['-', '--']

    split = 'Train'
    metric_names = ['F-score', 'Precision', 'Recall']

    event_based_df, segment_based_df = conf.load_train_results(model_name=model_name)
    prominence_threshold = 0.0
    coverage_threshold   = 0.5

    df = segment_based_df[segment_based_df['model_name'] == model_name]
    df = df[df['prominence_threshold'] == prominence_threshold]
    df = df[df['coverage_threshold'] == coverage_threshold]

    fig, ax = plt.subplots(n_rows, n_columns, figsize=figsize_side_by_side)
    for idx_metric, metric_name in enumerate(['f_measure']):
        for idx_noise, noise in enumerate([0.0, 0.2]):
            for idx_strat, strategy_name in enumerate(['OPT', 'ADP', 'CPD', 'FIX']):
                
                _df = df[df['strategy_name'] == strategy_name]
                _df = _df[_df['fn_noise'] == noise]
                _df = _df[_df['fp_noise'] == noise]
                average_df = _df.groupby(['n_queries'])[metric_name].mean().reset_index()

                # plot n_queries vs. f1-score
                title = r'Annotation quality ($\beta = {}$)'.format(noise) #'Dataset = {}'.format(split)

                ax[idx_noise].set_title(title)
                ax[idx_noise].set_xlabel(r'Queries per soundscape ($B$)')

                if idx_noise == 0:
                    ax[idx_noise].set_ylabel(metric_names[idx_metric])

                if strategy_name == 'OPT':
                    ax[idx_noise].plot(average_df['n_queries'][2:], average_df[metric_name][2:], '-o', color=colors[idx_strat], label='{}'.format(new_strategy_names[idx_strat]))
                else:
                    ax[idx_noise].plot(average_df['n_queries'], average_df[metric_name], '-o', color=colors[idx_strat], label='{}'.format(new_strategy_names[idx_strat]))
                ax[idx_noise].set_xticks(average_df['n_queries'], [str(x) for x in average_df['n_queries']])

                min_value = 0.10
                max_value = 0.57
                ax[idx_noise].set_ylim(min_value, max_value)
                
    ax[0].set_xlim(2.5, 13.5)
    ax[1].set_xlim(2.5, 13.5)

    alpha=0.15
    ax[0].fill_between([7, 15], min_value, max_value, alpha=alpha, color='grey')
    ax[1].fill_between([7, 15], min_value, max_value, alpha=alpha, color='grey')
    plt.tight_layout()
    ax[1].legend()
    plt.savefig(os.path.join(fig_dir, 'figure_3.pdf'), format='pdf', dpi=1200)


if __name__ == '__main__':
    main()
