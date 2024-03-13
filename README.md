# A-CPD: From Weak to Strong Sound Event Labels using Adaptive Change-Point Detection and Active Learning

![Figure 2](results/figures/figure_2.png)

- __TODO__: update with arXiv link (soon).
- __TODO__: update with Zenodo links (soon).
- __TODO__: update with proper citation (soon).

Official PyTorch implementation of the A-CPD method presented in the paper [From Weak to Strong Sound Event Labels using Adaptive Change-Point Detection and Active Learning](https://arxiv.org), by [John Martinsson](https://johnmartinsson.github.io), [Olof Mogren](https://mogren.one), [Maria Sandsten](https://www.maths.lu.se/english/research/staff/mariasandsten/), and [Tuomas Virtanen](https://homepages.tuni.fi/tuomas.virtanen/)

Currently under review for EUSIPCO 2024. Cite as:

    @article{Martinsson2024,
      title={From Weak to Strong Sound Event Labels using Adaptive Change-Point Detection and Active Learning},
      author={Martinsson, John and Mogren, Olof and Sandsten, Maria and Virtanen, Tuomas},
      journal={arXiv preprint arXiv:...},
      year={2024}
    }

## Setup environment and download datasets

    bash doit.sh

Please read the doit.sh file.

## Reproduce figures and tables using pre-computed result files

    # produces all tables in the paper
    python src/tables.py

    # produces all figures in the paper
    python src/figures.py

The figures are saved to the directory

    ./results/figures_reproduced

## Reproduce figures and tables using re-computed result files

    python src/main.py

This will run all experiments presented in the paper and store the results in,

    ./results/eusipco_2024_reproduced

However, only 2 runs are made per configuration to save time since the standard devaition is so low. Results should be similar, but may vary slightly.

Please update the config.py script after this and change

    results_dir : eusipco_2024 # to
    results_dir : eusipco_2024_reproduced

If you do not change this line, the produced figures and tables will be from the pre-computed eusipco_2024 results. Then run

    # produces all tables in the paper
    python src/tables.py

    # produces all figures in the paper
    python src/figures.py

## Reproduce figures and tables using re-generated audio datasets
A description on how to download all the audio source material and how to use the scripts to generate the datasets and compute the embeddings using BirdNET will be made available upon demand. Please contact the main author of the paper.
