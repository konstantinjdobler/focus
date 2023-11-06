from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from deepfocus import FOCUS

source_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
source_model: PreTrainedModel = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base")

target_tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
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
    # fasttext_model_path="/path/to/fasttext.bin", # or directly provide path to token-level fasttext model 
    
    # In the paper, we use `target_training_data_path` but we also implement using
    # WECHSEL's word-to-subword mapping if the language has pretrained fasttext word embeddings available online
    # To use, supply a two-letter `language_identifier` (e.g. "de" for German) instead of `target_training_data_path`
    # language_identifier="de",
)
source_model.resize_token_embeddings(len(target_tokenizer))
source_model.get_input_embeddings().weight.data = target_embeddings

# Continue training the model on the target language with `target_tokenizer`.
# ...
