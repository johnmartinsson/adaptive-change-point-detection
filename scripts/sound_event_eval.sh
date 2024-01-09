python sound_event_eval.py --class_name=baby --model_name=prototypical --sim_dir=results/10_runs_noisy_oracle_0.75_acc/baby/n_queries_7_fp_noise_0.25_fn_noise_0.25/ --t_collar=2.0 --n_runs=10
python sound_event_eval.py --class_name=baby --model_name=mlp --sim_dir=results/10_runs_noisy_oracle_0.75_acc/baby/n_queries_7_fp_noise_0.25_fn_noise_0.25/ --t_collar=2.0 --n_runs=10

python sound_event_eval.py --class_name=me --model_name=prototypical --sim_dir=results/10_runs_noisy_oracle_0.75_acc/me/n_queries_7_fp_noise_0.25_fn_noise_0.25/ --t_collar=2.0 --n_runs=10
python sound_event_eval.py --class_name=me --model_name=mlp --sim_dir=results/10_runs_noisy_oracle_0.75_acc/me/n_queries_7_fp_noise_0.25_fn_noise_0.25/ --t_collar=2.0 --n_runs=10

python sound_event_eval.py --class_name=dog --model_name=prototypical --sim_dir=results/10_runs_noisy_oracle_0.75_acc/dog/n_queries_7_fp_noise_0.25_fn_noise_0.25/ --t_collar=2.0 --n_runs=10
python sound_event_eval.py --class_name=dog --model_name=mlp --sim_dir=results/10_runs_noisy_oracle_0.75_acc/dog/n_queries_7_fp_noise_0.25_fn_noise_0.25/ --t_collar=2.0 --n_runs=10
