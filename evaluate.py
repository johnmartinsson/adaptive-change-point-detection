import os
import numpy as np

import utils
import metrics
import oracles
import datasets
import query_strategies as qs

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
    assert tot >= soundscape_length, "expected sum: {}, output sum: {}".format(soundscape_length, tot)
    assert tot <= soundscape_length + 0.5, "expected sum: {}, output sum: {}".format(soundscape_length, tot)

def evaluate_query_strategy(base_dir, soundscape_basename, query_strategy, min_iou=0.001, n_queries=0):
    query_strategy.base_dir = base_dir
    # create oracle
    oracle = oracles.WeakLabelOracle(base_dir)

    # create queries
    queries = query_strategy.predict_queries(soundscape_basename, n_queries)

    valid_queries(queries, base_dir, soundscape_basename, n_queries)

    pos_ref  = datasets.load_pos_ref_aux(base_dir, soundscape_basename)
    pos_pred = oracle.pos_events_from_queries(queries, soundscape_basename)
        
    f1_score       = metrics.f1_score_from_events(pos_ref, pos_pred, min_iou=min_iou)
    mean_iou_score = metrics.average_matched_iou(pos_ref, pos_pred, min_iou=min_iou)

    p_embeddings, n_embeddings = get_embeddings_2(pos_pred, base_dir, soundscape_basename)

    return f1_score, mean_iou_score, p_embeddings, n_embeddings

def get_embeddings_2(pos_pred, base_dir, soundscape_basename):                                                                           
    timings, embeddings = datasets.load_timings_and_embeddings(base_dir, soundscape_basename)                                            
    
    avg_timings = np.mean(timings, axis=1)                                                                                               
    
    p_embeddings = []                                                                                                                    
    n_embeddings = []                                                                                                                    
    
    idx_timing = 0 
    idx_pos_pred = 0                                                                                                                     
    not_done = True                                                                                                                      
    while not_done:
        s, e = pos_pred[idx_pos_pred]                                                                                                    
        idx_pos_pred += 1                                                                                                                
        
        # add negative embeddings
        while timings[idx_timing][1] < s:
            #print("{:.2f} negative".format(avg_timings[idx_timing]))
            n_embeddings.append(embeddings[idx_timing])                                                                                  
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
            idx_timing += 1                                                                                                              
        
        not_done = idx_pos_pred < len(pos_pred)                                                                                          
    
    # TODO: these if statements are a bit ad hoc                                                                                         
    if idx_timing < len(embeddings):
        rest_embeddings = embeddings[idx_timing:]                                                                                        
        if len(n_embeddings) > 0:
            n_embeddings = np.concatenate((np.array(n_embeddings), rest_embeddings)) # Add rest                                          
        else:
            n_embeddings = rest_embeddings                                                                                               

    p_embeddings = np.array(p_embeddings)
    n_embeddings = np.array(n_embeddings)
    
    return p_embeddings, n_embeddings 

def get_embeddings(pos_pred, base_dir, soundscape_basename):
    # TODO: this may actually introduce a lot of label-noise
    # make sure that all embeddings that contain a whole positive
    # event gets a positive label...
    timings, embeddings = datasets.load_timings_and_embeddings(base_dir, soundscape_basename)

    avg_timings = np.mean(timings, axis=1)

    p_embeddings = []
    n_embeddings = []

    idx_timing = 0
    idx_pos_pred = 0
    not_done = True
    while not_done:
        s, e = pos_pred[idx_pos_pred]
        idx_pos_pred += 1

        # add negative embeddings
        while avg_timings[idx_timing] < s:
            n_embeddings.append(embeddings[idx_timing])
            idx_timing += 1

        # add positive embeddings
        while avg_timings[idx_timing] < e:
            p_embeddings.append(embeddings[idx_timing])
            idx_timing += 1

        not_done = idx_pos_pred < len(pos_pred)

    p_embeddings = np.array(p_embeddings)
    # TODO: these if statements are a bit ad hoc
    if idx_timing < len(embeddings):
        rest_embeddings = embeddings[idx_timing:]
        if len(n_embeddings) > 0:
            n_embeddings = np.concatenate((np.array(n_embeddings), rest_embeddings)) # Add rest
        else:
            n_embeddings = rest_embeddings


    return p_embeddings, n_embeddings
