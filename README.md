# FOCUS

Code for FOCUS: Effective Embedding Initialization for Specializing Pretrained Multilingual Models on a Single Language

Preprint on arXiv: https://arxiv.org/abs/2305.14481.

## Usage

You can clone the repository or simply copy the `focus` folder. The necessary dependencies are listed in `requirements.txt` (`pip install -r requirements.txt`).

The following example shows how to use FOCUS to specialize `xlm-roberta-base` on German with a custom, language-specific tokenizer. The code is also available in [`focus_example.py`](focus_example.py).

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer
from focus import FOCUS

source_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
source_model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base")

target_tokenizer = AutoTokenizer.from_pretrained(
    "./tokenizers/de/xlm-roberta-base-50k"
)

# Example for training a new tokenizer:
# target_tokenizer = source_tokenizer.train_new_from_iterator(
#     load_dataset("cc100", lang="de", split="train")["text"],
#     vocab_size=50_432
# )
# target_tokenizer.save_pretrained("./target_tokenizer_test")

target_embeddings = FOCUS(
    source_embeddings=source_model.get_input_embeddings().weight,
    source_tokenizer=source_tokenizer,
    target_tokenizer=target_tokenizer,
    target_training_data_path="/path/to/data.txt"
    # In the paper, we use `target_training_data_path` but we also implement using
    # WECHSEL's word-to-subword mapping if the language has pretrained fasttext word embeddings available online
    # To use, supply a two-letter `language_identifier` (e.g. "de" for German) instead of `target_training_data_path`
    # language_identifier="de",
)
source_model.resize_token_embeddings(len(target_tokenizer))
source_model.get_input_embeddings().weight.data = target_embeddings

# Continue training the model on the target language with `target_tokenizer`.
# ...
```

## Citation

You can cite FOCUS like this:

```bibtex
@misc{dobler-demelo-2023-focus,
    title={FOCUS: Effective Embedding Initialization for Specializing Pretrained Multilingual Models on a Single Language},
    author={Konstantin Dobler and Gerard de Melo},
    year={2023},
    eprint={2305.14481},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

If you use the "WECHSEL-style" word-to-subword mapping, please consider also citing their [original work](https://github.com/CPJKU/wechsel).
