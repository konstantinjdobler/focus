# FOCUS

Code for "FOCUS: Effective Embedding Initialization for Monolingual Specialization of Multilingual Models" accepted at the EMNLP 2023 main conference.

Paper on arXiv: https://arxiv.org/abs/2305.14481.

## Installation

We provide the package via `pip install deepfocus`.

Alternatively, you can simply copy the `deepfocus` folder and drop it into your project.
The necessary dependencies are listed in `requirements.txt` (`pip install -r requirements.txt`).

## Usage

The following example shows how to use FOCUS to specialize `xlm-roberta-base` on German with a custom, language-specific tokenizer. The code is also available in [`focus_example.py`](focus_example.py).

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer
from deepfocus import FOCUS

source_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
source_model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base")

target_tokenizer = AutoTokenizer.from_pretrained(
    "./tokenizers/de/xlmr-unigram-50k"
)

# Example for training a new tokenizer:
# target_tokenizer = source_tokenizer.train_new_from_iterator(
#     load_dataset("cc100", lang="de", split="train")["text"],
#     vocab_size=50_048
# )
# target_tokenizer.save_pretrained("./target_tokenizer_test")

target_embeddings = FOCUS(
    source_embeddings=source_model.get_input_embeddings().weight,
    source_tokenizer=source_tokenizer,
    target_tokenizer=target_tokenizer,
    target_training_data_path="/path/to/data.txt"
    # fasttext_model_path="/path/to/fasttext.bin", # or directly provide path to token-level fasttext model 

    # In the paper, we use `target_training_data_path` but we also implement using
    # WECHSEL's word-to-subword mapping if the language has pretrained fasttext word embeddings available online
    # To use, supply a two-letter `language_identifier` (e.g. "de" for German) instead of `target_training_data_path` and set:
    # auxiliary_embedding_mode="fasttext-wordlevel",
    # language_identifier="de",

)
source_model.resize_token_embeddings(len(target_tokenizer))
source_model.get_input_embeddings().weight.data = target_embeddings

# Continue training the model on the target language with `target_tokenizer`.
# ...
```

## Checkpoints
We publish the checkpoints trained with FOCUS on HuggingFace:
| Language    | Vocabulary Replacement (preferred)                              | Vocabulary Extension                                 |
|-------------|-----------------------------------------------|------------------------------------------------|
| German      | [`konstantindobler/xlm-roberta-base-focus-german`](https://huggingface.co/konstantindobler/xlm-roberta-base-focus-german)           | [`konstantindobler/xlm-roberta-base-focus-extend-german`](https://huggingface.co/konstantindobler/xlm-roberta-base-focus-extend-german)          |
| Arabic      | [`konstantindobler/xlm-roberta-base-focus-arabic`](https://huggingface.co/konstantindobler/xlm-roberta-base-focus-arabic)           | [`konstantindobler/xlm-roberta-base-focus-extend-arabic`](https://huggingface.co/konstantindobler/xlm-roberta-base-focus-extend-arabic)          |
| Kiswahili   | [`konstantindobler/xlm-roberta-base-focus-kiswahili`](https://huggingface.co/konstantindobler/xlm-roberta-base-focus-kiswahili)     | [`konstantindobler/xlm-roberta-base-focus-extend-kiswahili`](https://huggingface.co/konstantindobler/xlm-roberta-base-focus-extend-kiswahili)|
| Hausa       | [`konstantindobler/xlm-roberta-base-focus-hausa`](https://huggingface.co/konstantindobler/xlm-roberta-base-focus-hausa)           | [`konstantindobler/xlm-roberta-base-focus-extend-hausa`](https://huggingface.co/konstantindobler/xlm-roberta-base-focus-extend-hausa)          |
| isiXhosa    | [`konstantindobler/xlm-roberta-base-focus-isixhosa`](https://huggingface.co/konstantindobler/xlm-roberta-base-focus-isixhosa)     | [`konstantindobler/xlm-roberta-base-focus-extend-isixhosa`](https://huggingface.co/konstantindobler/xlm-roberta-base-focus-extend-isixhosa)|

In our experiments, full vocabulary replacement coupled with FOCUS outperformed extending XLM-R's original vocabulary, while also resulting in a smaller model and being faster to train.

## Citation

You can cite FOCUS like this:

```bibtex
@misc{dobler-demelo-2023-focus,
    title={FOCUS: Effective Embedding Initialization for Monolingual Specialization of Multilingual Models},
    author={Konstantin Dobler and Gerard de Melo},
    year={2023},
    eprint={2305.14481},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

If you use the "WECHSEL-style" word-to-subword mapping, please consider also citing their [original work](https://github.com/CPJKU/wechsel).
