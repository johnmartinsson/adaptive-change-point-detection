import os
import numpy as np

import utils
import metrics
import oracles
import datasets
import query_strategies as qs

def valid_queries(queries, base_dir, soundscape_basename):
    soundscape_length = qs.get_soundscape_length(base_dir, soundscape_basename)

    sorted_queries = sorted(queries, key = lambda x: x[0])

    # check no overlap
    for idx_query in range(len(sorted_queries)-1):
        q1 = sorted_queries[idx_query]
        q2 = sorted_queries[idx_query + 1]

        assert q1[1] <= q2[0], "overlapping queries for soundscape: {}".format(soundscape_basename)

    # check sums correctly
    tot = 0
    for (s, e) in sorted_queries:
        L = e-s
        tot += L

    assert tot == soundscape_length, "expected sum: {}, output sum: {}".format(soundscape_length, tot)

def evaluate_all(base_dir, soundscape_basename, proto_active_learner, min_iou=0.001, n_queries=0):

    # create oracle
    oracle = oracles.WeakLabelOracle(base_dir)

    soundscape_length = qs.get_soundscape_length(base_dir, soundscape_basename)

    # create queries
    opt_queries = qs.optimal_query_strategy(base_dir, soundscape_basename, soundscape_length)
    if n_queries == 0:
        n_queries   = len(opt_queries) # this is the annotation budget
    cp_queries  = qs.change_point_query_strategy(n_queries, base_dir, soundscape_basename, soundscape_length)
    fix_queries = qs.fixed_query_strategy(n_queries, base_dir, soundscape_basename, soundscape_length)
    al_queries = proto_active_learner.predict_queries(soundscape_basename, n_queries)

    assert(len(cp_queries) == len(fix_queries) == len(al_queries))
    #print("fix queries: ", len(fix_queries))
    #print("AL queries: ", len(al_queries))

    # assert valied queries
    # TODO: some files have overlapping positive events, not as expected
    #valid_queries(opt_queries, base_dir, soundscape_basename)
    valid_queries(cp_queries, base_dir, soundscape_basename)
    valid_queries(fix_queries, base_dir, soundscape_basename)

    # TODO: I should activate this again for fair budget comparison
    #valid_queries(al_queries, base_dir, soundscape_basename)

    queries_and_names = [
        #(opt_queries, 'opt'),
        #(cp_queries,  'cp'),
        (fix_queries, 'fix'),
        (al_queries, 'al'),
    ]

    name_to_label_quality = {'f1':{}, 'iou':{}}
    ref_path = os.path.join(base_dir, '{}.txt'.format(soundscape_basename))
    
    #print("###################################################")
    for (queries, name) in queries_and_names:
        pos_ref  = datasets.load_pos_ref(ref_path)
        pos_pred = oracle.pos_events_from_queries(queries, soundscape_basename)
        
        label_quality_1 = metrics.f1_score_from_events(pos_ref, pos_pred, min_iou=min_iou)
        label_quality_2 = metrics.average_matched_iou(pos_ref, pos_pred, min_iou=min_iou)

        if False: #not name == 'opt' and not name == 'cp':
            print("-------------------------------------------")
            utils.print_queries(pos_ref, 'ref')
            utils.print_queries(pos_pred, 'pred')

            print("method: {}, soundscape: {}, f1-score: {}, mean iou: {}".format(name, soundscape_basename, label_quality_1, label_quality_2))

        #label_quality = label_quality_1 1 * label_quality_2

        name_to_label_quality['f1'][name] = label_quality_1
        name_to_label_quality['iou'][name] = label_quality_2

        if name == 'al':

            #print("------------------------------------------")
            #print("- ", soundscape_basename)
            #print("------------------------------------------")
            #print("pos_pred : ", ['({:.2f}, {:.2f})'.format(c[0], c[1]) for c in pos_pred])
            #print("pos_ref  : ", ['({:.2f}, {:.2f})'.format(c[0], c[1]) for c in pos_ref])

            p_embeddings, n_embeddings = get_embeddings(pos_pred, base_dir, soundscape_basename)

            #print("positive: ", p_embeddings.shape)
            #print("negative: ", n_embeddings.shape)

    return name_to_label_quality, len(opt_queries), p_embeddings, n_embeddings

def evaluate_query_strategy(base_dir, soundscape_basename, query_strategy, min_iou=0.001, n_queries=0):
    # create oracle
    oracle = oracles.WeakLabelOracle(base_dir)

    #soundscape_length = qs.get_soundscape_length(base_dir, soundscape_basename)

    # create queries
    queries = query_strategy.predict_queries(soundscape_basename, n_queries)

    valid_queries(queries, base_dir, soundscape_basename)

    ref_path = os.path.join(base_dir, '{}.txt'.format(soundscape_basename))
    pos_ref  = datasets.load_pos_ref(ref_path)
    pos_pred = oracle.pos_events_from_queries(queries, soundscape_basename)
        
    f1_score = metrics.f1_score_from_events(pos_ref, pos_pred, min_iou=min_iou)
    mean_iou_score = metrics.average_matched_iou(pos_ref, pos_pred, min_iou=min_iou)

    p_embeddings, n_embeddings = get_embeddings(pos_pred, base_dir, soundscape_basename)

    return f1_score, mean_iou_score, p_embeddings, n_embeddings

def get_embeddings_2(pos_pred, base_dir, soundscape_basename):
    soundscape_length = qs.get_soundscape_length(base_dir, soundscape_basename)
    neg_pred = datasets.compute_neg_from_pos(pos_pred, soundscape_length)

    timings, embeddings = datasets.load_timings_and_embeddings(base_dir, soundscape_basename)

    n_embeddings = []
    p_embeddings = []

    for idx in range(len(timings)):
        (s_emb, e_emb) = timings[idx]
        emb = embeddings[idx]

        for (s_pos, e_pos) in pos_pred:
            if s_pos >= s_emb and e_pos <= e_emb:
                p_embeddings.append(emb)
                continue

        for (s_neg, e_neg) in neg_pred:
            if s_neg <= s_emb and e_neg >= e_emb:
                n_embeddings.append(emb)
                continue

    n_embeddings = np.array(n_embeddings)
    p_embeddings = np.array(p_embeddings)

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
