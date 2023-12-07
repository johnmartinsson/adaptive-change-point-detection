import os
import numpy as np
import glob

import warnings

import metrics
import oracles
import datasets
import query_strategies as qs
import models

import argparse

def get_positive_annotations(fp):
    with open(fp, 'r') as f:
        lines = f.readlines()
        lines    = [line.split('\t') for line in lines[1:]]
        anns = [(float(s), float(e)) for (s, e, _) in lines]
    return anns

def valid_queries(queries, base_dir, soundscape_basename, n_queries):
    soundscape_length = qs.get_soundscape_length(base_dir, soundscape_basename)

    sorted_queries = sorted(queries, key = lambda x: x[0])

    # check no overlap
    for idx_query in range(len(sorted_queries)-1):
        q1 = sorted_queries[idx_query]
        q2 = sorted_queries[idx_query + 1]

        assert q1[1] <= q2[0], "overlapping queries for soundscape: {}".format(soundscape_basename)

    # check budget is respected
    assert len(queries) <= n_queries, "the budget is not respected."

    # check sums correctly
    tot = 0
    for (s, e) in sorted_queries:
        L = e-s
        tot += L

    # TODO: hacky add of 0.5 and check less than because of embeddings in BirdNET...
    assert tot <= soundscape_length, "expected sum: {}, output sum: {}".format(soundscape_length, tot)
    #assert tot <= soundscape_length + 0.6, "expected sum: {}, output sum: {}".format(soundscape_length, tot)

def evaluate_query_strategy(base_dir, soundscape_basename, query_strategy, min_iou=0.001, n_queries=0, noise_factor=0, normalize=False, iteration=0, emb_win_length=1.0, fp_noise=0.0, fn_noise=0.0):
    #query_strategy.base_dir = base_dir
    # create oracle
    oracle = oracles.WeakLabelOracle(base_dir, fp_noise=fp_noise, fn_noise=fn_noise)

    # create queries
    queries = query_strategy.predict_queries(base_dir, soundscape_basename, n_queries, noise_factor=noise_factor, normalize=normalize, iteration=iteration)

    valid_queries(queries, base_dir, soundscape_basename, n_queries)

    pos_ref  = datasets.load_pos_ref_aux(base_dir, soundscape_basename)
    pos_pred = oracle.pos_events_from_queries(queries, soundscape_basename)

    #assert len(pos_pred) <= 3, "either oracle is wrong, or there are more than 3 events."
        
    # TODO: this is unlikely to happen, but can happen if all positive events end up overlapping with two queries by change.
    if len(pos_pred) == 0:
        f1_score = 0
        mean_iou_score = 0
        warnings.warn('Unlikely behaviour for {}, no positive labels from oracle, may skew results ...'.format(soundscape_basename))
        print("pos_pred: ", pos_pred)
        print("pos_ref: ", pos_ref)
        print("queries: ", queries)
    else:
        f1_score       = metrics.f1_score_from_events(pos_ref, pos_pred, min_iou=min_iou)
        mean_iou_score = metrics.average_matched_iou(pos_ref, pos_pred, min_iou=min_iou)

    # TODO: changed this now
    #p_embeddings, n_embeddings = get_embeddings_2(pos_pred, base_dir, soundscape_basename)

    # TODO: set emb window length according to the actually used window length !!!!!!
    p_embeddings, n_embeddings, _ = get_embeddings_3(pos_pred, base_dir, soundscape_basename, emb_win_length=emb_win_length)

    return f1_score, mean_iou_score, p_embeddings, n_embeddings, pos_pred

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
    timings, embeddings = datasets.load_timings_and_embeddings(base_dir, soundscape_basename)                                            
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

    return p_embs, n_embs, embs_label #, idx_pos_embs, idx_neg_embs

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

def predict_test_data(query_strategy, base_dir, scores_dir, emb_win_length, class_name):
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
        probas = query_strategy.predict_probas(embeddings)

        # Event-based predictions for collar evaluation
        pos_indices = (probas >= 0.5)
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim_dir', help='The directory to save the results in', required=True, type=str)
    parser.add_argument('--class_name', required=True, type=str)
    parser.add_argument('--emb_win_length', required=True, type=float)
    parser.add_argument('--strategy_name', required=True, type=str)
    parser.add_argument('--n_runs', required=True, type=int)
    args = parser.parse_args()

    emb_win_length = args.emb_win_length
    emb_hop_length = emb_win_length / 4

    emb_hop_length_str = '{:.2f}'.format(emb_hop_length)
    emb_win_length_str = '{:.1f}'.format(emb_win_length)
    class_name = args.class_name

    train_base_dir = '/mnt/storage_1/datasets/bioacoustic_sed/generated_datasets/{}_{}_{}s/train_soundscapes_snr_0.0'.format(class_name, emb_win_length_str, emb_hop_length_str)

    for idx_run in range(args.n_runs):
        print("run = ", idx_run)
        train_annotation_dir   = os.path.join(args.sim_dir, args.strategy_name, str(idx_run), 'train_annotations')

        # load train annotations
        train_annotation_paths = glob.glob(os.path.join(train_annotation_dir, "*.tsv"))

        def get_iteration(fp):
            return int(os.path.basename(fp).split('_')[1])

        def get_soundscape_basename(fp):
            return "_".join(os.path.basename(fp).split('_')[2:]).split('.')[0]
        
        evaluation_budgets = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.0]
        n_soundscapes      = np.max([get_iteration(fp) for fp in train_annotation_paths]) + 1
        n_iters            = [int(evaluation_budget * n_soundscapes) for evaluation_budget in evaluation_budgets]

        for idx_budget, n_iter in enumerate(n_iters):
            # 1. load the annotations until n_iter
            budget_train_annotation_paths = [fp for fp in train_annotation_paths if get_iteration(fp) < n_iter]
            soundscape_basenames          = [get_soundscape_basename(fp) for fp in budget_train_annotation_paths]

            # 2. create model using these annotations
            p_embss   = []
            n_embss   = []
            for idx, soundscape_basename in enumerate(soundscape_basenames):

                #pos_ann = np.load(budget_train_annotation_paths[idx])
                pos_ann = get_positive_annotations(budget_train_annotation_paths[idx])
                p_embs, n_embs, _ = get_embeddings_3(pos_ann, train_base_dir, soundscape_basename, args.emb_win_length)
                p_embs = np.array(p_embs)
                n_embs = np.array(n_embs)

                p_embss.append(p_embs)
                n_embss.append(n_embs)

            p_embs = np.concatenate(p_embss)
            n_embs = np.concatenate(n_embss)
            
            # NOTE: we only use the predictive model, never the queries, i.e, the query strategies do not matter here
            query_strategy = models.AdaptiveQueryStrategy(train_base_dir, random_soundscape=False, fixed_queries=False, emb_cpd=False, normal_prototypes=True)
            # update the model with the annotated data
            query_strategy.update(p_embs, n_embs)

            # 3. evaluate the model on the test data
            test_base_dir = train_base_dir.replace('train', 'test')
            f1_test_score, miou_test_score = evaluate_model_on_test_data(query_strategy, test_base_dir)

            print("strategy {}, budget {}, f1 = {:.3f}, miou = {:.3f} (test)".format(args.strategy_name, evaluation_budgets[idx_budget], f1_test_score, miou_test_score))            
        
            # 4. predict the test data, and save to disk
            scores_dir = os.path.join(args.sim_dir, args.strategy_name, str(idx_run), 'test_scores', 'budget_{}'.format(evaluation_budgets[idx_budget]))
            predict_test_data(query_strategy, test_base_dir, scores_dir, args.emb_win_length, args.class_name)

if __name__ == '__main__':
    main()
