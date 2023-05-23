from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from focus import FOCUS

source_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
source_model: PreTrainedModel = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base")

target_tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
    "./tokenizers/de/xlm-roberta-base-50k"
)

# Example for training a new tokenizer
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
    # alternative if language has pretrained fasttext word embedding available online and you do not want to provide `target_training_data_path`:
    # language_identifier="de",
)
source_model.resize_token_embeddings(len(target_tokenizer))
source_model.get_input_embeddings().weight.data = target_embeddings

# Continue training the model with target_tokenizer
# ...
