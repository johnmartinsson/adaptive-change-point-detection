from ray import tune

def development(labeling_budget):

    search_space = {
        'labeling_budget' : labeling_budget,
        'batch_size' : tune.grid_search([16]),
    }

    return search_space
