import numpy as np
import sys
import os

result_dir = sys.argv[1]

f1_scores_train   = np.load(os.path.join(result_dir, "f1_scores_train.npy"))
miou_scores_train = np.load(os.path.join(result_dir, "miou_scores_train.npy"))

n_strategies = f1_scores_train.shape[0]
n_query_settings = f1_scores_train.shape[1]

n_queriess = [5,7,10,20,30]

for idx_n_queries in range(n_query_settings):
    n_queries = n_queriess[idx_n_queries]

    print("------------------------------------------------")
    print("- Number of queries: {}".format(n_queries))
    print("------------------------------------------------")
    for idx_query_strategy in range(n_strategies):
        f1_mean_train = f1_scores_train[idx_query_strategy, idx_n_queries].flatten().mean()
        f1_std_train  = f1_scores_train[idx_query_strategy, idx_n_queries].flatten().std()

        miou_mean_train = miou_scores_train[idx_query_strategy, idx_n_queries].flatten().mean()
        miou_std_train  = miou_scores_train[idx_query_strategy, idx_n_queries].flatten().std()
        print("Strategy {}, f1 = {:.3f} +- {:.3f}, miou = {:.3f} +- {:.3f}".format(idx_query_strategy, f1_mean_train, f1_std_train, miou_mean_train, miou_std_train))


