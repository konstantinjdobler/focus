import pytest
import transformers
from transformers import AutoTokenizer


@pytest.fixture(scope="session")
def xlmr_embeddings():
    return transformers.AutoModelForMaskedLM.from_pretrained("xlm-roberta-base").get_input_embeddings().weight


@pytest.fixture(scope="session")
def xlmr_tokenizer():
    return AutoTokenizer.from_pretrained("xlm-roberta-base", use_fast=True)


@pytest.fixture(scope="session")
def de_tokenizer():
    return AutoTokenizer.from_pretrained("./tokenizers/de/xlmr-unigram-50k", use_fast=True)


@pytest.fixture(scope="session")
def de_extend_tokenizer():
    return AutoTokenizer.from_pretrained("./tokenizers/de/xlmr-unigram-30k", use_fast=True)


@pytest.fixture(scope="session")
def de_extended_tokenizer():
    return AutoTokenizer.from_pretrained("./tokenizers/de/xlm-roberta-base-extended", use_fast=True)
