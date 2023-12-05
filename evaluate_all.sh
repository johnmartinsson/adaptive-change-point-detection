python evaluate.py --results_dir=results_test/ --name=$1 --emb_win_length=1.0 --strategy_name=OPT --class_name=$1
python evaluate.py --results_dir=results_test/ --name=$1 --emb_win_length=1.0 --strategy_name=ADP --class_name=$1
python evaluate.py --results_dir=results_test/ --name=$1 --emb_win_length=1.0 --strategy_name=CPD --class_name=$1
python evaluate.py --results_dir=results_test/ --name=$1 --emb_win_length=1.0 --strategy_name=FIX --class_name=$1

# python evaluate.py --results_dir=results_test/ --name=dog --emb_win_length=1.0 --strategy_name=OPT --class_name=dog
# python evaluate.py --results_dir=results_test/ --name=dog --emb_win_length=1.0 --strategy_name=ADP --class_name=dog
# python evaluate.py --results_dir=results_test/ --name=dog --emb_win_length=1.0 --strategy_name=CPD --class_name=dog
# python evaluate.py --results_dir=results_test/ --name=dog --emb_win_length=1.0 --strategy_name=FIX --class_name=dog

# python evaluate.py --results_dir=results_test/ --name=me --emb_win_length=1.0 --strategy_name=OPT --class_name=me
# python evaluate.py --results_dir=results_test/ --name=me --emb_win_length=1.0 --strategy_name=ADP --class_name=me
# python evaluate.py --results_dir=results_test/ --name=me --emb_win_length=1.0 --strategy_name=CPD --class_name=me
# python evaluate.py --results_dir=results_test/ --name=me --emb_win_length=1.0 --strategy_name=FIX --class_name=me