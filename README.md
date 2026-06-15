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

Model Naming
---------
The model naming follows this format
```
<encoder_name>_<cmd_decoder_name>_<args_decoder_name>
```

- `<encoder_name>`: Encoder name or architecture (e.g. `t5`, `bert`)
- `<cmd_decoder_name>`: CMD decoder name or architecture (e.g. `t5`, `torch`)
- `<args_decoder_name>`: Args decoder name or architecture (e.g. `t5`, `torch`), or `cmdonly` if the model is trained on command sequence only

For class names, use CamelCase equivalents of each segment (e.g. `t5` → `T5`, `torch` → `Torch`, `cmdonly` → `Cmdonly`).

Examples
--------
`T5T5T5` (`t5_t5_t5.py`)
- T5 Encoder
- T5 CMD Decoder
- T5 Args Decoder
- Trained on both command and argument sequences

`T5T5Cmdonly` (`t5_t5_cmdonly.py`)
- T5 Encoder
- T5 CMD Decoder
- Trained on command sequence only

`T5TorchTorch` (`t5_torch_torch.py`)
- T5 Encoder
- Torch CMD Decoder
- Torch Args Decoder
- Trained on both command and argument sequences

`T5TorchCmdonly` (`t5_torch_cmdonly.py`)
- T5 Encoder
- Torch CMD Decoder
- Trained on command sequence only



References
---------

Khan, M. S., Sinha, S., Sheikh, T. U., Stricker, D., Ali, S. A., & Afzal, M. Z. (2024). Text2CAD: Generating sequential CAD models from beginner-to-expert level text prompts. arXiv. https://doi.org/10.48550/arXiv.2409.17106

<!-- If you use this project, please cite it as:

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

This project builds on open research and community tools. See module headers for attributions. -->
