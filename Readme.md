prjkcad

Overview
--------

`prjkcad` is a small CAD-data research toolkit for experimenting with text-to-CAD and geometry pipelines. It provides utilities for dataset handling, model training wrappers, lightweight visualization, and export helpers. The codebase is organized into `cadlib/`, `models/`, `utils/`, and `render/` for clear separation of concerns.

Installation
------------

1. Create a Python 3.8+ virtual environment and activate it:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

Usage
-----

Run training by injecting a config file from the `configs/` folder into the training utility `utils.train_model`.

Python API example:

```python
from utils.train_model import main as train_main

# path to one of the example configs, e.g. configs/experiment_1a.yaml
config_path = 'configs/experiment_1a.yaml'

train_main(config_path)
```

Command-line example (if supported):

```bash
python -m utils.train_model configs/experiment_1a.yaml
```

Adjust the config file to set dataset paths, model selection, and training hyperparameters. See the `configs/` folder for templates.

Reference
---------

Khan, et al. (2024). One relevant match found online is a systematic review titled "Global insights and the impact of generative AI-ChatGPT on multidisciplinary: a systematic review and bibliometric analysis" by N. Khan and co-authors (Taylor & Francis, 2024). If this is the paper you meant, its DOI is `10.1080/09540091.2024.2353630` and the publisher page is: https://www.tandfonline.com/doi/abs/10.1080/09540091.2024.2353630

- Khan, N., Koubaa, A., MK Khan, et al. (2024). Global insights and the impact of generative AI-ChatGPT on multidisciplinary: a systematic review and bibliometric analysis. Taylor & Francis. DOI: 10.1080/09540091.2024.2353630

How to cite this repo
---------------------

If you use this project, please cite it as:

Alif Amirudin (2026). prjkcad. GitHub repository: REPLACE_WITH_REPO_URL

BibTeX example (replace URL and year as appropriate):

```bibtex
@misc{amirudin_prjkcad_2026,
	author = {Alif Amirudin},
	title = {prjkcad},
	year = {2026},
	howpublished = {\url{REPLACE_WITH_REPO_URL}}
}
```

Acknowledgements
----------------

This project builds on open research and community tools. See module headers for attributions.

Contact
-------

For questions, contact Alif Amirudin.
