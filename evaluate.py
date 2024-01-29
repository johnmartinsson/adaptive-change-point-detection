import os
import numpy as np
import glob
import sys

import sklearn.svm
import sklearn.ensemble
import sklearn.linear_model
import sklearn.neighbors

import warnings

import metrics
import oracles
import datasets
import query_strategies as qs
import models
import evaluation_models

import argparse

def get_positive_annotations(fp):
    with open(fp, 'r') as f:
        lines = f.readlines()
        lines    = [line.split('\t') for line in lines[1:]]
        anns = [(float(s), float(e)) for (s, e, _) in lines]
    return anns

def valid_queries(queries, base_dir, soundscape_basename, n_queries, opt_queries):
    soundscape_length = qs.get_soundscape_length(base_dir, soundscape_basename)

    sorted_queries = sorted(queries, key = lambda x: x[0])

    # check no overlap
    for idx_query in range(len(sorted_queries)-1):
        q1 = sorted_queries[idx_query]
        q2 = sorted_queries[idx_query + 1]

        assert q1[1] <= q2[0], "overlapping queries for soundscape: {}".format(soundscape_basename)

    # check budget is respected
    if not opt_queries:
        assert len(queries) <= n_queries, "the budget is not respected: {}/{} queries.".format(len(queries), n_queries)

    # check sums correctly
    tot = 0
    for (s, e) in sorted_queries:
        L = e-s
        tot += L

    # TODO: hacky add of 0.5 and check less than because of embeddings in BirdNET...
    #assert tot <= soundscape_length, "expected sum: {}, output sum: {}".format(soundscape_length, tot)
    assert tot <= soundscape_length + 0.001, "expected sum: {}, output sum: {}".format(soundscape_length, tot)

def evaluate_query_strategy(base_dir, soundscape_basename, query_strategy, min_iou=0.001, n_queries=0, noise_factor=0, normalize=False, iteration=0, emb_win_length=1.0, fp_noise=0.0, fn_noise=0.0, prominence_threshold=0.0, converage_threshold=0.0):
    #query_strategy.base_dir = base_dir
    # create oracle
    oracle = oracles.WeakLabelOracle(base_dir, fp_noise=fp_noise, fn_noise=fn_noise, coverage_threshold=converage_threshold)

    # create queries
    queries = query_strategy.predict_queries(base_dir, soundscape_basename, n_queries, prominence_threshold=prominence_threshold, noise_factor=noise_factor, normalize=normalize, iteration=iteration)

    valid_queries(queries, base_dir, soundscape_basename, n_queries, query_strategy.opt_queries)

    pos_ref  = datasets.load_pos_ref_aux(base_dir, soundscape_basename)
    pos_pred = oracle.pos_events_from_queries(queries, soundscape_basename)

    #assert len(pos_pred) <= 3, "either oracle is wrong, or there are more than 3 events."
        
    # TODO: this is unlikely to happen, but can happen if all positive events end up overlapping with two queries by change.
    if len(pos_pred) == 0:
        f1_score = 0
        mean_iou_score = 0
        #warnings.warn('Unlikely behaviour for {}, no positive labels from oracle, may skew results ...'.format(soundscape_basename))
        #print("pos_pred: ", pos_pred)
        #print("pos_ref: ", pos_ref)
        #print("queries: ", queries)
    else:
        f1_score       = metrics.f1_score_from_events(pos_ref, pos_pred, min_iou=min_iou)
        mean_iou_score = metrics.average_matched_iou(pos_ref, pos_pred, min_iou=min_iou)

    # TODO: changed this now
    #p_embeddings, n_embeddings = get_embeddings_2(pos_pred, base_dir, soundscape_basename)

    # TODO: set emb window length according to the actually used window length !!!!!!
    p_embeddings, n_embeddings, _ = get_embeddings_3(pos_pred, base_dir, soundscape_basename, emb_win_length=emb_win_length)

    return f1_score, mean_iou_score, p_embeddings, n_embeddings, pos_pred, len(queries)

def get_embeddings_2(pos_pred, base_dir, soundscape_basename, emb_win_length):
    timings, embeddings = datasets.load_timings_and_embeddings(base_dir, soundscape_basename)                                            
    
    avg_timings = np.mean(timings, axis=1)                                                                                               
    
    p_embeddings = []                                                                                                                    
    n_embeddings = []                                                                                                                    
    
    idx_timing = 0 
    idx_pos_pred = 0                                                                                                                     
    not_done = True                                                                                                                      

    embs_label = np.zeros(len(timings))-1

    # if there are no positive annotations, everything contributes to the negative
    # TODO: the fact that this happens should be looked into
    if len(pos_pred) == 0:
        return [], np.array(embeddings)

    while not_done:
        s, e = pos_pred[idx_pos_pred]                                                                                                    
        idx_pos_pred += 1                                                                                                                
        
        # add negative embeddings
        while timings[idx_timing][1] < s:
            #print("{:.2f} negative".format(avg_timings[idx_timing]))
            n_embeddings.append(embeddings[idx_timing])                                                                                  
            embs_label[idx_timing] = 0
            idx_timing += 1                                                                                                              
        
        # ignore embeddings which may overlap                                                                                            
        while avg_timings[idx_timing] < s:
            #print("{:.2f} not used".format(avg_timings[idx_timing]))
            idx_timing += 1                                                                                                              
        
        # add positive embeddings if center-point of embedding is inside positive event timings                                          
        #print(base_dir)
        while idx_timing < len(timings) and avg_timings[idx_timing] <= e:
            #print("{}, {:.2f} positive embedding".format(soundscape_basename, avg_timings[idx_timing]))
            p_embeddings.append(embeddings[idx_timing])                                                                                  
            embs_label[idx_timing] = 1
            idx_timing += 1                                                                                                              
        
        not_done = idx_pos_pred < len(pos_pred)                                                                                          
    
    while idx_timing < len(timings):
        #print("{:.2f} negative".format(avg_timings[idx_timing]))
        n_embeddings.append(embeddings[idx_timing])                                                                                  
        embs_label[idx_timing] = 0
        idx_timing += 1                                                                                                              

    return p_embeddings, n_embeddings, embs_label

def get_embeddings_3(pos_ann, base_dir, soundscape_basename, emb_win_length):
    timings, embeddings = datasets.load_timings_and_embeddings(base_dir, soundscape_basename, normalize=True)                                            
    taus = np.mean(timings, axis=1)
    soundscape_length = qs.get_soundscape_length(base_dir, soundscape_basename)
    neg_ann =  datasets.compute_neg_from_pos(pos_ann, soundscape_length)

    idx_pos_embs = np.zeros(len(embeddings)) == 1
    idx_neg_embs = np.zeros(len(embeddings)) == 1
    embs_label   = np.zeros(len(embeddings))-1

    # TODO: maybe there should be a coverage criterion instead? E.g. how much of the positive annotation that is covered?
    L = emb_win_length
    for idx, tau in enumerate(taus):
        emb_q = (tau-L/2, tau+L/2)
        for (a_s, a_e) in pos_ann:
            ann_q = (a_s, a_e)
            if tau - L/2 >= a_s and tau + L/2 <= a_e:
                #print("1: ", emb_q, ann_q)
                idx_pos_embs[idx] = True
                embs_label[idx] = 1
            # TODO: how do I do this choice justly?
            if metrics.coverage(ann_q, emb_q) >= 0.001:
                #print("1: ", emb_q, ann_q)
                idx_pos_embs[idx] = True
                embs_label[idx] = 1
        for (a_s, a_e) in neg_ann:
            ann_q = (a_s, a_e)
            if tau - L/2 >= a_s and tau + L/2 <= a_e:
                #print("0: ", emb_q, ann_q)
                idx_neg_embs[idx] = True
                embs_label[idx] = 0

    n_embs = embeddings[idx_neg_embs]
    p_embs = embeddings[idx_pos_embs]

    return p_embs, n_embs, embs_label

# TODO: include this in default loop
def evaluate_annotation_process_on_test_data(query_strategy, base_dir, n_queries, noise_factor, fp_noise=0.0, fn_noise=0.0):
    soundscape_basenames = [os.path.basename(b).split('.')[0] for b in glob.glob(os.path.join(base_dir, "*.wav"))]

    f1s   = []
    mious = []

    oracle = oracles.WeakLabelOracle(base_dir, fp_noise=0.0, fn_noise=0.0)

    for soundscape_basename in soundscape_basenames:
        ref_pos  = datasets.load_pos_ref_aux(base_dir, soundscape_basename)

        queries = query_strategy.predict_queries(base_dir, soundscape_basename, n_queries, noise_factor=noise_factor)
        pred_pos = oracle.pos_events_from_queries(queries, soundscape_basename)

        if not len(pred_pos) == 0:
            f1   = metrics.f1_score_from_events(ref_pos, pred_pos, min_iou=0.00000001)
            miou = metrics.average_matched_iou(ref_pos, pred_pos, min_iou=0.00000001)
            f1s.append(f1)
            mious.append(miou)
        else:
            warnings.warn("No predictions, results will potentially be skewed ...")
            print("query strategy, fixed = {}, CPD = {}".format(query_strategy.fixed_queries, query_strategy.emb_cpd))
            print("pos_pred: ", pred_pos)
            print("pos_ref: ", ref_pos)
            print("queries: ", queries)
            # TODO: not sure, strong penalization of no predictions
            f1s.append(0)
            mious.append(0)
            
    return np.mean(f1s), np.mean(mious)

def predict_test_data(model, base_dir, scores_dir, emb_win_length, class_name):
    if not os.path.exists(scores_dir):
        os.makedirs(scores_dir)

    if not os.path.exists(os.path.join(scores_dir, 'event_based')):
        os.makedirs(os.path.join(scores_dir, 'event_based'))
    
    if not os.path.exists(os.path.join(scores_dir, 'segment_based')):
        os.makedirs(os.path.join(scores_dir, 'segment_based'))

    #print(base_dir)
    soundscape_basenames = [os.path.basename(b).split('.')[0] for b in glob.glob(os.path.join(base_dir, "*.wav"))]
    for soundscape_basename in soundscape_basenames:
        timings, embeddings  = datasets.load_timings_and_embeddings(base_dir, soundscape_basename)
        probas = model.predict_proba(embeddings)

        # TODO: this is a hack, but it works for now
        if len(probas.shape) == 2:
            probas = probas[:,1]

        # Event-based predictions for collar evaluation
        pos_indices     = (probas >= 0.5)
        avg_timings     = timings.mean(axis=1)
        pos_avg_timings = avg_timings[pos_indices]
        hop_length      = avg_timings[1]-avg_timings[0]

        pos_events = []
        idx_timing = 0
        # TODO: maybe improve this a bit? fairly naive as it is
        while idx_timing < len(pos_avg_timings):
            s = pos_avg_timings[idx_timing]
            # keep incrementing until we are at the end, or there is a gap in the predictions
            while idx_timing < len(pos_avg_timings)-1 and (pos_avg_timings[idx_timing+1] - pos_avg_timings[idx_timing]) <= hop_length:
                idx_timing += 1
            e = pos_avg_timings[idx_timing]
            pos_events.append((s, e))
            idx_timing += 1
        
        row = '{}\t{}\t{}'
        with open(os.path.join(scores_dir, 'event_based', '{}.txt'.format(soundscape_basename)), 'w') as proba_f:
            for (s, e) in pos_events:
                row_str = row.format(s, e, class_name)
                proba_f.write(row_str + '\n')

        # Segment-based predictions (without overlap) for PSDS evaluation
        taus = np.mean(timings, axis=1)
        header = 'onset\toffset\t{}'.format(class_name)
        row = '{}\t{}\t{}'
        with open(os.path.join(scores_dir, 'segment_based', '{}.tsv'.format(soundscape_basename)), 'w') as proba_f:
            proba_f.write(header + '\n')
            for idx in range(len(taus)):
                # TODO: PSDS does not allow overlapping events, so we only use every 4th embedding
                # this will probably affect the recall of most methods
                # NOTE: I am correcting the onset and offset here based on the rectangular window, 
                # so that the PSDS evaluation is correct.
                if idx % 4 == 0:
                    tau    = taus[idx]
                    onset  = tau - (emb_win_length / 2)
                    offset = tau + (emb_win_length / 2)
                    p      = probas[idx]
                    row_str = row.format(onset, offset, p)

                    proba_f.write(row_str + '\n')

def predict_train_data(sim_dir, base_dir, class_name, method_name, idx_run, emb_win_length=1.0):
    # print the class name, method name, and run index
    # to stdout so we can see the progress, and then flush
    sys.stdout.write("Class: {}, Method: {}, Run: {}\n".format(class_name, method_name, idx_run))
    sys.stdout.flush()
    #print("Class: {}, Method: {}, Run: {}".format(class_name, method_name, idx_run))
            
    run_dir                     = os.path.join(sim_dir, str(idx_run))
    train_soundscape_file_paths = glob.glob(os.path.join(base_dir, '*.wav'))
    train_annotation_file_paths = glob.glob(os.path.join(run_dir, 'train_annotations', '*.tsv'))
    train_scores_dir            = os.path.join(run_dir, 'train_scores')

    if not os.path.exists(train_scores_dir):
        os.makedirs(train_scores_dir)
    if not os.path.exists(os.path.join(train_scores_dir, 'event_based')):
        os.makedirs(os.path.join(train_scores_dir, 'event_based'))
    if not os.path.exists(os.path.join(train_scores_dir, 'segment_based')):
        os.makedirs(os.path.join(train_scores_dir, 'segment_based'))


    def get_soundscape_basename(fp):
        return os.path.splitext(os.path.basename(fp))[0]

    def get_soundscape_id(fp):
        return os.path.basename(fp).split('_')[-1].split('.')[0]

    for train_soundscape_file_path in train_soundscape_file_paths:
        train_sounscape_basename = get_soundscape_basename(train_soundscape_file_path)
        train_annotation_file    = [fp for fp in train_annotation_file_paths if get_soundscape_id(fp) == get_soundscape_id(train_sounscape_basename)][0]
        
        # TODO: pre-load embeddings and timings, takes ~0.01s
        timings, embeddings = datasets.load_timings_and_embeddings(base_dir, train_sounscape_basename)
        pos_ann = get_positive_annotations(train_annotation_file)
        
        # TODO: pre-load embedding labels, takes ~0.01s
        _, _, embs_label = get_embeddings_3(pos_ann, base_dir, train_sounscape_basename, emb_win_length)
        taus = np.mean(timings, axis=1)
        window_timings = [(tau - emb_win_length / 2, tau + emb_win_length / 2) for tau in taus]

        idx_nonoverlapping = np.arange(len(window_timings)) % 4 == 0
        window_timings_nonoverlapping = np.array(window_timings)[idx_nonoverlapping]
        embs_label_nonoverlapping     = embs_label[idx_nonoverlapping]
        
        # Event-based for collar eval
        with open(os.path.join(train_scores_dir, 'event_based', train_sounscape_basename + '.txt'), 'w') as f:
            for (s, e) in pos_ann:
                f.write('{}\t{}\t{}\n'.format(s, e, class_name))

        # Segment-based for PSDS eval
        with open(os.path.join(train_scores_dir, 'segment_based', train_sounscape_basename + '.tsv'), 'w') as f:
            f.write('onset\toffset\t{}\n'.format(class_name))
            for (s, e), l in zip(window_timings_nonoverlapping, embs_label_nonoverlapping):
                f.write('{}\t{}\t{}\n'.format(s, e, l))


def evaluate_model_on_test_data(query_strategy, base_dir, threshold=0.5):
    soundscape_basenames = [os.path.basename(b).split('.')[0] for b in glob.glob(os.path.join(base_dir, "*.wav"))]

    f1s   = []
    mious = []
    for soundscape_basename in soundscape_basenames:
        ref_pos  = datasets.load_pos_ref_aux(base_dir, soundscape_basename)
        pred_pos = query_strategy.predict_pos_events(base_dir, soundscape_basename, threshold=threshold)
        if not len(pred_pos) == 0:
            f1   = metrics.f1_score_from_events(ref_pos, pred_pos, min_iou=0.00000001)
            miou = metrics.average_matched_iou(ref_pos, pred_pos, min_iou=0.00000001)
            f1s.append(f1)
            mious.append(miou)
        else:
            # TODO: not sure, strong penalization of no predictions
            f1s.append(0)
            mious.append(0)
            
    return np.mean(f1s), np.mean(mious)

def predict_test_and_train(conf):
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--sim_dir', help='The directory to save the results in', required=True, type=str)
    # parser.add_argument('--class_name', required=True, type=str)
    # parser.add_argument('--emb_win_length', required=True, type=float)
    # parser.add_argument('--strategy_name', required=True, type=str, help='The name of the annotation strategy to evaluate')
    # parser.add_argument('--model_name', required=True, type=str, help='The name of the model to evaluate')
    # parser.add_argument('--n_runs', required=True, type=int)
    # parser.add_argument('--base_dir', required=True, type=str)
    # parser.add_argument('--only_budget_1', required=True, type=str)
    # args = parser.parse_args()

    #train_base_dir = conf.train_base_dir #'{}/generated_datasets/{}_{}_{}s/train_soundscapes_snr_0.0'.format(args.base_dir, class_name, emb_win_length_str, emb_hop_length_str)

    for idx_run in range(conf.n_runs):
        train_annotation_dir   = os.path.join(conf.sim_dir, str(idx_run), 'train_annotations')
        #print(train_annotation_dir)

        # load train annotations
        train_annotation_paths = glob.glob(os.path.join(train_annotation_dir, "*.tsv"))

        def get_iteration(fp):
            return int(os.path.basename(fp).split('_')[1])

        def get_soundscape_basename(fp):
            return "_".join(os.path.basename(fp).split('_')[2:]).split('.')[0]

        # TODO: something goes wrong here with budget 1.0, max over empty list
        # the problem is most likely solved, the n_runs were inconsistent with the number of runs in the sim_dir
        #print(train_annotation_paths)
        n_soundscapes      = np.max([get_iteration(fp) for fp in train_annotation_paths]) + 1
        n_iters            = [int(evaluation_budget * n_soundscapes) for evaluation_budget in conf.evaluation_budgets]

        for idx_budget, n_iter in enumerate(n_iters):
            #print("strategy = {}, run = {}, model_name = {}, budget = {}".format(args.strategy_name, idx_run, args.model_name, conf.evaluation_budgets[idx_budget]))
            sys.stdout.write("strategy = {}, run = {}, model_name = {}, budget = {}, n_iter = {}\n".format(conf.strategy_name, idx_run, conf.model_name, conf.evaluation_budgets[idx_budget], n_iter))
            # 1. load the annotations until n_iter
            budget_train_annotation_paths = [fp for fp in train_annotation_paths if get_iteration(fp) < n_iter]
            #print(budget_train_annotation_paths)
            assert len(budget_train_annotation_paths) == n_iter, "budget not respected, expected {}, got {}".format(n_iter, len(budget_train_annotation_paths))
            soundscape_basenames          = [get_soundscape_basename(fp) for fp in budget_train_annotation_paths]

            # 2. load embeddings and annotations for current budget
            p_embss   = []
            n_embss   = []
            for idx, soundscape_basename in enumerate(soundscape_basenames):

                #pos_ann = np.load(budget_train_annotation_paths[idx])
                pos_ann = get_positive_annotations(budget_train_annotation_paths[idx])
                p_embs, n_embs, _ = get_embeddings_3(pos_ann, conf.train_base_dir, soundscape_basename, conf.emb_win_length)
                p_embs = np.array(p_embs)
                n_embs = np.array(n_embs)

                p_embss.append(p_embs)
                n_embss.append(n_embs)

            # positive and negative embeddings
            p_embs = np.concatenate(p_embss)
            n_embs = np.concatenate(n_embss)

            # TODO: batch normalization of embeddings in MLP seems to improve performance a lot,
            # is it possible that the embeddings are not normalized correctly?
            #print("mean p_embs: ", np.mean(p_embs, axis=0))
            #print("std p_embs: ",  np.std(p_embs, axis=0))
            
            # 3. train the model using the annotated embeddings
            if conf.model_name == 'prototypical':
                # NOTE: we only use the predictive model, never the queries, i.e, the query strategies do not matter here
                model = models.AdaptiveQueryStrategy(conf) #train_base_dir, random_soundscape=False, fixed_queries=False, emb_cpd=False, normal_prototypes=True)
                # update the model with the annotated data
                model.update(p_embs, n_embs)
            elif conf.model_name == 'linear_svm':
                model = sklearn.svm.SVC(kernel='linear', probability=True) #, verbose=True, max_iter=5000)
            elif conf.model_name == 'rbf_svm':
                model = sklearn.svm.SVC(kernel='rbf', probability=True) #, verbose=True, max_iter=5000)
            elif conf.model_name == 'random_forest':
                model = sklearn.ensemble.RandomForestClassifier(n_estimators=100)
            elif conf.model_name == 'logistic_regression':
                model = sklearn.linear_model.LogisticRegression(max_iter=5000)
            elif conf.model_name == 'knn':
                model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=5, weights='distance')
            elif conf.model_name == 'mlp':
                model = evaluation_models.MLPClassifier(1024, 256, 2, verbose=False, max_epochs=500, patience=10, batch_size=64)
            else:
                raise ValueError("Unknown model name: {}".format(conf.model_name))
            
            X = np.concatenate([p_embs, n_embs])
            y = np.concatenate([np.ones(len(p_embs)), np.zeros(len(n_embs))])
            if not conf.model_name == 'prototypical':
                # TODO: this is a hack to make it run faster
                if 'svm' in conf.model_name:
                    X = X[:10000]
                    y = y[:10000]
                model.fit(X, y)

            #test_base_dir = train_base_dir.replace('train', 'test')
            # the directory to save the prediction probas in
            scores_dir = os.path.join(conf.sim_dir, str(idx_run), 'test_scores', conf.model_name, 'budget_{}'.format(conf.evaluation_budgets[idx_budget]))

            # 5. predict the test data, and save to disk
            predict_test_data(model, conf.test_base_dir, scores_dir, conf.emb_win_length, conf.class_name)

            # 6. predict scores for the train data, and save to disk
            # Note: the annotation process only provides event timings,
            # which means that the score is 1 if the embedding is inside,
            # and 0 otherwise. Basically, this converts the training annotations
            # to a format suitable for sed_eval and sed_scores_eval
            predict_train_data(conf.sim_dir, conf.train_base_dir, conf.class_name, conf.strategy_name, idx_run, conf.emb_win_length)

            sys.stdout.flush()

# if __name__ == '__main__':
#     main()
