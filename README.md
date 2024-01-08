# Active learning for bioacoustics

A repository containing an active learning framework for bioacoustics.

# Prerequisites

## Data
TODO: add the doi:s and links to the datasets.

    - NIGENS dataset,
    - TUT Rare Events dataset,
    - DCASE Few-shot bioacoustic dataset,

In the ``produce_source_material.py'' you need to set the correct data paths:

        tut_base_dir    = '<path>/TUT_rare_sed_2017/TUT-rare-sound-events-2017-development/data/source_data/'
        nigens_base_dir = '<path>/NIGENS/'
        dcase_base_dir  = '<path>/Development_Set/Validation_Set/ME/'

## Environment
Check the requirements.txt for the requirements. In particular we need:

    - Scaper, and
    - BirdNET-Analyser.

Scaper is used to generate the soundscapes using the source data, and BirdNET-Analyser is used to pre-compute the embeddings that the method works on.

# Doit.sh

If everything is setup properly you should be able to run everything by simply writing:

        bash doit.sh

If this does not work, it can at least be used as documentation on the order in which scripts need to be run. Basically:

    - data generation scripts,
    - active learning simulation scripts,
    - model training and prediction scripts, and
    - evaluation of annotations and the predictions at test time.

# Generate the dataset

TODO: describe how to generate the data in more detail.


# Extract the embeddings

TODO: explain how to generate the embeddings in more detail.

Setup the BirdNET-Analyzer v2.4 (https://github.com/kahst/BirdNET-Analyzer).

    python embeddings.py --i ./scaper_source_files/BV/AMRE/train_soundscapes/ --o ./scaper_source_files/BV/AMRE/train_soundscapes/ --threads 8 --batchsize 16 --overlap 0
    python embeddings.py --i ./scaper_source_files/BV/AMRE/train_soundscapes/ --o ./scaper_source_files/BV/AMRE/train_soundscapes/ --threads 8 --batchsize 16 --overlap 0

This will generate the embeddings for the train_soundscapes and the test_soundscapes and store them in the respective directory. Embeddings are for 3 second segments with the specified overlap.
    
# Run simulations

TODO: explain how to run the simulations in more detail.

# Run predictions

# Run evaluation

# Produce figures and tables