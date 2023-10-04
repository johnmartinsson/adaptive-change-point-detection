def load_pos_ref(ref_path):
    pos_ref = []
    
    with open(ref_path, 'r') as f:
        ls = f.readlines()
        for l in ls:
            ds = l.split('\t')
            start_time = float(ds[0])
            end_time = float(ds[1])
            pos_ref.append((start_time, end_time))
    return pos_ref
    
def load_neg_ref(ref_path, soundscape_length):
    pos_ref = load_pos_ref(ref_path)

    neg_ref = []
    prev_pos_end_time   = 0
    for (curr_pos_start_time, curr_pos_end_time) in pos_ref:
        neg_start_time = prev_pos_end_time
        neg_end_time   = curr_pos_start_time
        neg_ref.append((neg_start_time, neg_end_time))
    
        prev_pos_end_time   = curr_pos_end_time
    
    if prev_pos_end_time < soundscape_length:
        neg_ref.append((prev_pos_end_time, soundscape_length))

    return neg_ref
