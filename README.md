# AND-GloRe

***A**uthor **N**ame **D**isambiguation via **Glo**bal and **Re**fined Views*

## Overview

🚧 *Work in progress...*

## Data

Run **AND-GloRe** experiments on the following AND datasets:

- WhoIsWho-v1
  - **Release date:** 2019
  - **Source:** https://cn.aminer.org/open/article?id=5de9efd2530c707ed8b87d99 

- WhoIsWho-v2
  - **Release date:** 2020
  - **Source:** https://cn.aminer.org/open/article?id=5de9efd2530c707ed8b87d99


To generate the Heterogeneous Information Networks (HINs) for each ambiguous group in the datasets, two specific files are required in the `data/<dataset_name>` folder.

1. `preprocessed.csv`

  Preprocessed dataset CSV file containing:
    - `id`: publication identifier.
    - `split`: dataset split for the ambiguous-name group, e.g., `test`.
    - `name`: normalized ambiguous name used to group rows into one HIN.
    - `author`: ground-truth author/person identifier for the publication.
    - `label`: integer ground-truth label for author within its `name` group.
    - `title`: preprocessed publication title text.
    - `abstract`: preprocessed publication abstract text.
    - `keywords`: list-like string of publication keywords.
    - `authors`: list-like string of normalized paper author names.
    - `orgs`: list-like string of author organizations/affiliations, aligned by position with `authors`.
    - `venue`: preprocessed publication venue name.
    - `year`: publication year, when available.

2. `features.pt`

    PyTorch-serialized dictionary with the following format:

    ```python
    {
        "<publication_id>": torch.Tensor  # shape: (embedding_dim,)
    }
    ```

Both files for each dataset can be downloaded from [releases](https://github.com/Victorgonl/AND-GloRe/releases/datasets/) and extracted into the `data` folder.

## Running

Install local `andglore` package and all dependencies found in `requirements.txt`.

```
pip install -e .
```

Generate the HINs for a dataset:

```
python scripts/generate_networks.py --config configs/whoiswhov1.yaml
```

Run the experiment with parameters specified in the `.yaml` config file.

```
python scripts/run_experiment.py --config configs/whoiswhov1.yaml
```

Experiment logs are printed to the terminal, and more detailed logs are stored in the `logs` folder.

Experiment metrics are stored in the `results` folder.

Disambiguation representations and inferred clusters are stored in the `outputs` folder.

### Running the MCCG baseline

MCCG uses the same `preprocessed.csv` and `features.pt` files. First generate its
homogeneous paper networks, then run the reference MCCG training method:

```
python scripts/generate_networks_mccg.py --config configs/whoiswhov1_mccg.yaml
python scripts/run_experiment_mccg.py --config configs/whoiswhov1_mccg.yaml
```

The optional coauthor filters used by AND-GloRe network generation are also
available for MCCG:

```
python scripts/generate_networks_mccg.py \
  --config configs/whoiswhov1_mccg.yaml \
  --max_orgs_per_author 10 \
  --max_author_paper_ratio 0.5
```

MCCG networks, logs, results, outputs, scripts, and configurations use the
`_mccg` suffix so they do not overwrite AND-GloRe artifacts.

## Citation

🚧 *Work in progress...*
