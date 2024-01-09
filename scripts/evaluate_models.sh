
#bash ./scripts/evaluate_all.sh $1 results/10_runs_noisy_oracle_0.75_acc/$1/n_queries_7_fp_noise_0.0_fn_noise_0.0/ random_forest
#bash ./scripts/evaluate_all.sh $1 results/10_runs_noisy_oracle_0.75_acc/$1/n_queries_7_fp_noise_0.0_fn_noise_0.0/ logistic_regression
#bash ./scripts/evaluate_all.sh $1 results/10_runs_noisy_oracle_0.75_acc/$1/n_queries_7_fp_noise_0.0_fn_noise_0.0/ knn
bash ./scripts/evaluate_all.sh $1 results/10_runs_noisy_oracle_0.75_acc/$1/n_queries_7_fp_noise_0.25_fn_noise_0.25/ prototypical
bash ./scripts/evaluate_all.sh $1 results/10_runs_noisy_oracle_0.75_acc/$1/n_queries_7_fp_noise_0.25_fn_noise_0.25/ mlp