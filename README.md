# From Weak to Strong Sound Event Labels using Adaptive Change-Point Detection and Active Learning
TODO: add primary method image
TODO: remove or move unused scrips, currently you have to scroll for the README.


A repository containing an annotaion framework for bioacoustics


# Prerequisites

## Data

The dataset can be downloaded from: https://github.com/johnmartinsson/bioacoustics-task

TODO: write a script which downloads this and puts it into a default data folder

TODO: upload the pre-computed embeddings.

## Environment
Check the requirements.txt for the requirements. In particular we need:

    - Scaper, and
    - BirdNET-Analyser.

Scaper is used to generate the soundscapes using the source data, and BirdNET-Analyser is used to pre-compute the embeddings that the method works on.

## Pre-compute the embeddings

TODO: last part of 

    scripts/generate_scaper_data.sh

# Run simulations / experiments

TODO: update main script with proper default results folder

If everything is setup properly you should be able to run everything by simply writing:

        python main.py

this should run

- active learning annotation simulation,
- model training and prediction, and
- evaluation of models trained with annotations.

# Produce figures and tables

    # produces all tables in the paper
    python tables.py

    # produces all figures in the paper
    python figures.py

# Generate the dataset

TODO: describe how to generate the data in more detail.

## Produce source files
TODO: add the doi:s and links to the datasets.

    - NIGENS dataset,
    - TUT Rare Events dataset,
    - DCASE Few-shot bioacoustic dataset,

In the ``produce_source_material.py'' you need to set the correct data paths:

        tut_base_dir    = '<path>/TUT_rare_sed_2017/TUT-rare-sound-events-2017-development/data/source_data/'
        nigens_base_dir = '<path>/NIGENS/'
        dcase_base_dir  = '<path>/Development_Set/Validation_Set/ME/'

## Generate audio recordings

TODO:


## Extract the embeddings

TODO: explain how to generate the embeddings in more detail.

Setup the BirdNET-Analyzer v2.4 (https://github.com/kahst/BirdNET-Analyzer).

    python embeddings.py --i ./scaper_source_files/BV/AMRE/train_soundscapes/ --o ./scaper_source_files/BV/AMRE/train_soundscapes/ --threads 8 --batchsize 16 --overlap 0
    python embeddings.py --i ./scaper_source_files/BV/AMRE/train_soundscapes/ --o ./scaper_source_files/BV/AMRE/train_soundscapes/ --threads 8 --batchsize 16 --overlap 0

This will generate the embeddings for the train_soundscapes and the test_soundscapes and store them in the respective directory. Embeddings are for 3 second segments with the specified overlap.
    

