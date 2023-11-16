import string
from dataclasses import dataclass

import numpy as np
import regex
from torch import Tensor
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from .logger import logger


@dataclass
class TokenClass:
    native_form: str
    id: int
    canonical_form: str
    fuzzy_form: str
    uncased_form: str
    no_whitespace_form: str
    is_beginning_of_word: bool
    descriptor: str = ""


@dataclass
class NewToken:
    target: TokenClass
    auxiliary_embedding: Tensor | np.ndarray = None
    descriptor: str = ""


@dataclass
class OverlappingToken:
    source: list[TokenClass]
    target: TokenClass
    source_embedding: Tensor = None
    auxiliary_embedding: Tensor | np.ndarray = None
    descriptor: str = ""
    use_for_focus: bool = True


BPE_WHITESPACE = "Ġ"
XLMR_WHITESPACE = "▁"


def get_canonicalize_token_fn(input_tokenizer: PreTrainedTokenizer):
    """Standardize tokens from different tokenizers."""

    def decode(tokenizer: PreTrainedTokenizer, token_id: int):
        """For BPE tokenizer and fallback"""
        decoded_token = tokenizer.decode(token_id)
        token = tokenizer.convert_ids_to_tokens(token_id)
        is_beginning_of_word = token.startswith(BPE_WHITESPACE)
        if is_beginning_of_word:
            return XLMR_WHITESPACE + decoded_token.lstrip(), True
        else:
            return decoded_token.lstrip(), False

    def replace_space(tokenizer: PreTrainedTokenizer, token_id: int):
        """For XLM-R tokenizer (sentencepiece-style)"""
        decoded_token = tokenizer.decode(token_id)
        token = tokenizer.convert_ids_to_tokens(token_id)
        
        # For sentencepiece ByteFallback tokens used in Llama, Mistral et al.
        if regex.match(r"<0x[0-9,A-F]{2}>", token):
            return token, False

        is_beginning_of_word = token.startswith(XLMR_WHITESPACE)
        if is_beginning_of_word:
            return XLMR_WHITESPACE + decoded_token.lstrip(), True
        else:
            return decoded_token.lstrip(), False

    def wordpiece(tokenizer: PreTrainedTokenizer, token_id: int):
        """For wordpiece (e.g. BERT or mBERT)"""
        token = tokenizer.decode(token_id)
        if token.startswith("##"):
            return token[2:], False
        else:
            return XLMR_WHITESPACE + token, True

    # simple heuristics to avoid false positive
    if len([k for k in input_tokenizer.get_vocab().keys() if k[0] == XLMR_WHITESPACE]) > 100:
        logger.debug(f"Using sentencepiece canonicalization for {input_tokenizer}")
        return replace_space
    elif len([k for k in input_tokenizer.get_vocab().keys() if k[:2] == "##"]) > 100:
        logger.debug(f"Using wordpiece canonicalization for {input_tokenizer}")
        return wordpiece
    else:
        logger.debug(f"Using default canonicalization for {input_tokenizer}")
        return decode


def is_numerical_symbol_etc(token: str, tokenizer: PreTrainedTokenizer):
    if token in tokenizer.all_special_tokens:
        return True
    return token.isnumeric() or all(c in string.punctuation for c in token) or token.isspace()


def canonicalize_vocab(vocab, tokenizer, descriptor):
    canonical_vocab: dict[str, TokenClass] = {}
    canonicalize_token = get_canonicalize_token_fn(tokenizer)
    for token, token_idx in tqdm(vocab.items(), desc=f"Canonicalizing {descriptor} vocab", leave=False):
        canonical_form, is_beginning_of_word = canonicalize_token(tokenizer, token_idx)
        token_info = TokenClass(
            native_form=token,
            canonical_form=canonical_form,
            fuzzy_form=canonical_form.replace(XLMR_WHITESPACE, "").lower(),
            uncased_form=canonical_form.lower(),
            no_whitespace_form=canonical_form.replace(XLMR_WHITESPACE, ""),
            id=token_idx,
            is_beginning_of_word=is_beginning_of_word,
            descriptor=descriptor,
        )

        canonical_vocab[token] = token_info
    return canonical_vocab


def construct_vocab_view(vocab: dict[str, TokenClass], key: str):
    view: dict[str, list[TokenClass]] = {}

    # sort to ensure deterministic order.
    for token, token_info in sorted(vocab.items(), key=lambda x: x[1].id):
        cur_key_value = token_info.__getattribute__(key)
        if view.get(cur_key_value):
            if cur_key_value == token_info.__getattribute__("canonical_form"):
                # ensure canonical form is always first
                view[cur_key_value].insert(0, token_info)
            else:
                view[cur_key_value].append(token_info)
        else:
            view[cur_key_value] = [token_info]
    return view
