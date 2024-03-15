# A-CPD: From Weak to Strong Sound Event Labels using Adaptive Change-Point Detection and Active Learning

![Figure 2](results/figures/figure_2.png)

Official PyTorch implementation of the A-CPD method presented in the paper [From Weak to Strong Sound Event Labels using Adaptive Change-Point Detection and Active Learning](https://arxiv.org/abs/2403.08525), by [John Martinsson](https://johnmartinsson.github.io), [Olof Mogren](https://mogren.one), [Maria Sandsten](https://www.maths.lu.se/english/research/staff/mariasandsten/), and [Tuomas Virtanen](https://homepages.tuni.fi/tuomas.virtanen/)

[arXiv](https://arxiv.org/abs/2403.08525) | [Zenodo](https://zenodo.org/records/10811797)

Currently under review for EUSIPCO 2024. Cite as:

    @misc{Martinsson2024,
          title={From Weak to Strong Sound Event Labels using Adaptive Change-Point Detection and Active Learning}, 
          author={John Martinsson and Olof Mogren and Maria Sandsten and Tuomas Virtanen},
          year={2024},
          eprint={2403.08525},
          archivePrefix={arXiv},
          primaryClass={cs.SD}
    }

## Just do it
Please read the doit.sh file. This command requires ~17GB of free disk space.

    bash doit.sh

This will reproduce all figures and tables of the paper.

'./results/figures_reproduced'

If you are a bit more careful you could read through the doit.sh file and execute each command by itself.

## Reproduce figures and tables using reproduced experiment results

    python src/main.py

This will run all experiments presented in the paper and store the results in,

    ./results/eusipco_2024_reproduced

In the interest of time only 2 runs are done by default per configuration since the standard devaition is so low. All results should be similar, but may vary slightly.

Please update the src/config.py script after this and change to

    results_dir : eusipco_2024_reproduced

If you do not change this line, the produced figures and tables will be from the pre-computed eusipco_2024 results. Then run

    # produces all tables in the paper
    python src/tables.py

    # produces all figures in the paper
    python src/figures.py

## Reproduce figures and tables using reproduced audio mixture datasets
A description on how to download all the audio source material and how to use the scripts to generate the datasets and compute the embeddings using BirdNET will be made available upon demand. Please contact the main author of the paper.
