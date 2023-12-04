import os
import numpy as np
import glob

import warnings

import utils
import metrics
import oracles
import datasets
import query_strategies as qs

import argparse

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', help='The directory to save the results in', required=True, type=str)
    parser.add_argument('--name', required=True, type=str)
    parser.add_argument('--train_data_dir', required=True, type=str)
    parser.add_argument('--emb_win_length', required=True, type=float)
    parser.add_argument('--strategy_name', required=True, type=str)
    args = parser.parse_args()

    # implement so can be run over multiple runs and take average ...
    train_annotation_dir = os.path.join(args.results_dir, args.name, 'train_annotations', args.strategy_name, '0')
    train_annotation_paths = glob.glob(os.path.join(train_annotation_dir, "*.npy"))
    #print(train_annotation_paths)

    soundscape_basenames = [os.path.basename(fp).split('.')[0] for fp in train_annotation_paths]
    #print(soundscape_basenames)

    p_embss = []
    n_embss = []
    densities = []
    for idx, soundscape_basename in enumerate(soundscape_basenames):
        pos_ann = np.load(train_annotation_paths[idx])
        p_embs, n_embs, _ = get_embeddings_3(pos_ann, args.train_data_dir, soundscape_basename, args.emb_win_length)
        p_embs = np.array(p_embs)
        n_embs = np.array(n_embs)

        p_embss.append(p_embs)
        n_embss.append(n_embs)
        print(soundscape_basename, p_embs.shape, n_embs.shape)
        timings, embeddings = datasets.load_timings_and_embeddings(args.train_data_dir, soundscape_basename)
        soundscape_length = qs.get_soundscape_length(args.train_data_dir, soundscape_basename)
        neg_ann =  datasets.compute_neg_from_pos(pos_ann, soundscape_length)
        #print("timings: ", np.mean(timings, axis=1))
        #print("pos_ann: ", [('{:.2f}'.format(s), '{:.2f}'.format(e)) for (s, e) in pos_ann])
        #print("neg_ann: ", [('{:.2f}'.format(s), '{:.2f}'.format(e)) for (s, e) in neg_ann])

        density = np.sum([(e-s) for (s, e) in pos_ann]) / soundscape_length
        #print("density: ", density)
        densities.append(density)

    p_embs = np.concatenate(p_embss)
    n_embs = np.concatenate(n_embss)


    # train a classifier on these embeddings

    # evaluate the classifier on the test data

    # save the results to disk

    print("density: ", np.mean(densities))
    print("p_embs: ", p_embs.shape)
    print("n_embs: ", n_embs.shape)

    return

if __name__ == '__main__':
    main()
