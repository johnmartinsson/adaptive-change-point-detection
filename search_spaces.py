from ray import tune

def development(labeling_budget):

    search_space = {
        'labeling_budget' : labeling_budget,
        'batch_size' : tune.grid_search([16]),
        'segment_size' : 2000,
        'max_nb_segments' : 100,
    }

    return search_space
