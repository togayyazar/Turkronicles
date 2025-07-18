# Turkronicles

**Diachronic Resources for the Fast Evolving Turkish Language**

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Example Usage](#example-usage)
- [Citation](#citation)


## Overview

This repository hosts all code and data used in “Turkronicle: Diachronic Resources for the Fast-Evolving Turkish Language.” Inside, you’ll find decade-separeted corpora (1921-2024), pre-trained static embeddings (PPMI, SVD, CBOW), 1-,2-,3-,4-,and 5-grams with their frequencies. \
To access diachronic resources: https://zenodo.org/records/15766757.
Each reseource lives in its associated folder. 

## Installation of Lingan
Lingan makes the use of embeddings and ngrams easy by providing core functions for computational diachronic analysis such embedding alingment. Documentation for the code is not ready right now. Please refer to the Appendix section for the list of defined operations in the library.
Lingan can only be installed by cloning the repository for now. We are planning to upload the codebase to the pypi to support pip installation.

## Example Usage

See `lingan/examples/exps.py`

## Citation

If you use the resources presented in this repository, please cite:

@misc{yazar2024turkroniclesdiachronicresourcesfast,\
      title={Turkronicles: Diachronic Resources for the Fast Evolving Turkish Language},\ 
      author={Togay Yazar and Mucahid Kutlu and İsa Kerem Bayırlı},\
      year={2024},\
      eprint={2405.10133},\
      archivePrefix={arXiv},\
      primaryClass={cs.CL},\
      url={https://arxiv.org/abs/2405.10133},
}
