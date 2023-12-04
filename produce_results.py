import os
import numpy as np
import sys
import matplotlib.pyplot as plt

def get_miou_scores_test(result_dir):
    return np.load(os.path.join(result_dir, "miou_scores_test.npy"))

def get_f1_scores_test(result_dir):
    return np.load(os.path.join(result_dir, "f1_scores_test.npy"))

def get_f1_scores_train(result_dir):
    return np.load(os.path.join(result_dir, "f1_scores_train.npy"))

def get_miou_scores_train(result_dir):
    return np.load(os.path.join(result_dir, "miou_scores_train.npy"))

def get_miou_scores_train_online(result_dir):
    return np.load(os.path.join(result_dir, "miou_scores_train_online.npy"))

def get_f1_scores_train_online(result_dir):
    return np.load(os.path.join(result_dir, "f1_scores_train_online.npy"))

def plot_model_performance_on_test_data(result_dir):
    miou_scores_test = get_miou_scores_test(result_dir)
    f1_scores_test   = get_f1_scores_test(result_dir)

    strategy_names = ['OPT', 'ADP', 'CPD', 'FIX']

    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    for idx_strategy in [0, 1, 2, 3]:
        mean_mious = miou_scores_test[idx_strategy].mean(axis=(0, 1))
        mean_f1s   = f1_scores_test[idx_strategy].mean(axis=(0, 1))

        ax[0].plot(mean_mious, label=strategy_names[idx_strategy]) #, color=colors[idx_strategy])
        ax[0].set_xlabel('Annotated batches of soundscapes')
        ax[0].set_ylabel('mIOU')
        ax[0].legend(loc='lower right')
        ax[0].set_ylim([0, 1.05])
        
        ax[1].plot(mean_f1s, label=strategy_names[idx_strategy]) #, color=colors[idx_strategy])
        ax[1].set_xlabel('Annotated batches of soundscapes')
        ax[1].set_ylabel('F-score')
        ax[1].legend(loc='lower right')
        ax[1].set_ylim([0, 1.05])

    plt.suptitle('Model performance on test data')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'model_performance_on_test_data.png'), bbox_inches='tight')
    return ax

def plot_annotation_quality_on_test_data(result_dir):
    miou_scores_train = get_miou_scores_train(result_dir)
    f1_scores_train   = get_f1_scores_train(result_dir)

    strategy_names = ['OPT', 'ADP', 'CPD', 'FIX']

    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    for idx_strategy in [0, 1, 2, 3]:
        mean_mious = miou_scores_train[idx_strategy].mean(axis=(0, 1))
        mean_f1s   = f1_scores_train[idx_strategy].mean(axis=(0, 1))

        ax[0].plot(mean_mious, label=strategy_names[idx_strategy]) #, color=colors[idx_strategy])
        ax[0].set_xlabel('Annotated batches of soundscapes')
        ax[0].set_ylabel('mIOU')
        ax[0].legend(loc='lower right')
        ax[0].set_ylim([0, 1.05])
        
        ax[1].plot(mean_f1s, label=strategy_names[idx_strategy]) #, color=colors[idx_strategy])
        ax[1].set_xlabel('Annotated batches of soundscapes')
        ax[1].set_ylabel('F-score')
        ax[1].legend(loc='lower right')
        ax[1].set_ylim([0, 1.05])

    plt.suptitle('Annotation quality on test data')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'annotation_quality_on_test_data.png'), bbox_inches='tight')
    return ax

def table_online_annotation_quality(result_dir):
    miou_scores_train_online = get_miou_scores_train_online(result_dir)
    f1_scores_train_online   = get_f1_scores_train_online(result_dir)

    strategy_names = ['OPT', 'ADP', 'CPD', 'FIX']
    if "me" in result_dir:
        class_name = "meerkat"
    if "dog" in result_dir:
        class_name = "dog"
    if "baby" in result_dir:
        class_name = "baby"

    caption_text = "Training label quality on {} sounds.".format(class_name)
    print("\caption {}".format(caption_text))
    print("    & F-score                   & mIOU")
    print("\hline")
    for idx_query_strategy in [0, 1, 2, 3]:
        f1_mean   = f1_scores_train_online[idx_query_strategy].mean()
        f1_std    = f1_scores_train_online[idx_query_strategy].std()
        miou_mean = miou_scores_train_online[idx_query_strategy].mean()
        miou_std  = miou_scores_train_online[idx_query_strategy].std()

        print("{} & ${:.3f} \pm {:.3f}$ &  ${:.3f} \pm {:.3f}$".format(
            strategy_names[idx_query_strategy], f1_mean, f1_std, miou_mean, miou_std))
    print("\hline")

def main():
    # TODO
    sim_dir = sys.argv[1]
    plot_model_performance_on_test_data(sim_dir)
    plot_annotation_quality_on_test_data(sim_dir)
    print("")
    table_online_annotation_quality(sim_dir)

    #sim_dir = os.path.join(result_dir, 'dog_L=1.0_N=7')
    #plot_model_performance_on_test_data(sim_dir)
    #plot_annotation_quality_on_test_data(sim_dir)
    #print("")
    #table_online_annotation_quality(sim_dir)

    #sim_dir = os.path.join(result_dir, 'me_L=0.8_N=7')
    #plot_model_performance_on_test_data(sim_dir)
    #plot_annotation_quality_on_test_data(sim_dir)
    #print("")
    #table_online_annotation_quality(sim_dir)


if __name__ == '__main__':
    main()
