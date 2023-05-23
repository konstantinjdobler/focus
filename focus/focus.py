from collections import OrderedDict

import numpy as np
import torch
from loguru import logger
from torch import Tensor
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from .fasttext_embs import load_target_token_embedding

BPE_WHITESPACE = "Ġ"
XLMR_WHITESPACE = "▁"


def get_token_standardization_func(input_tokenizer: PreTrainedTokenizer):
    """Standardize tokens from different tokenizers.
    Standard output format should be Unicode-like output for non-ASCII chars.
    Beginning of word tokens should be prefixed with a space.

    We have to use .decode() to get "standardized" tokens (e.g. BytePairBPE represents non-ASCII tokens non-UNIcode-like internally).
    But XLM-R's tokenizer removes leading whitespace from tokens when using .decode().
    Se we add those back in manually.
    """

    def decode(tokenizer: PreTrainedTokenizer, token_id: int):
        """For BPE tokenizer and fallback"""
        return tokenizer.decode(token_id)

    def replace_space(tokenizer: PreTrainedTokenizer, token_id: int):
        """For XLM-R tokenizer (sentencepiece-style)"""
        return tokenizer.convert_ids_to_tokens(token_id).replace(XLMR_WHITESPACE, " ")

    def wordpiece(tokenizer: PreTrainedTokenizer, token_id: int):
        """For wordpiece (e.g. BERT or mBERT)"""
        token = tokenizer.decode(token_id)
        if token.startswith("##"):
            return token[2:]
        else:
            return " " + token

    # simple heuristics to avoid false positive
    if (
        len([k for k in input_tokenizer.get_vocab().keys() if k[0] == XLMR_WHITESPACE])
        > 100
    ):
        standardize_token = replace_space
    # simple heuristics to avoid false positive
    elif len([k for k in input_tokenizer.get_vocab().keys() if k[:2] == "##"]) > 100:
        standardize_token = wordpiece
    else:
        standardize_token = decode

    return standardize_token


def get_overlapping_tokens(
    target_tokenizer: PreTrainedTokenizer,
    source_tokenizer: PreTrainedTokenizer,
    fuzzy_search=True,
    fuzzy_whitespace=False,
):
    target_vocab = target_tokenizer.get_vocab()
    source_vocab = source_tokenizer.get_vocab()

    standardize_token = get_token_standardization_func(source_tokenizer)
    source_vocab = {
        standardize_token(source_tokenizer, idx): idx
        for idx in sorted(source_vocab.values())
    }

    standardize_token = get_token_standardization_func(target_tokenizer)
    target_vocab = {
        standardize_token(target_tokenizer, idx): idx
        for idx in sorted(target_vocab.values())
    }

    # Determine overlapping tokens between source and target vocab
    exact_overlap = {
        k: (target_vocab[k], source_vocab[k])
        for k in set(target_vocab) & set(source_vocab)
    }

    if not fuzzy_search:
        return {
            target_tokenizer.convert_ids_to_tokens(v[0]): v
            for k, v in sorted(exact_overlap.items())
        }

    # We do a greedy search for additional overlapping tokens.
    # NOTE: source_vocab order is random, need to sort for consistent results
    lowercase_source_vocab = {k.lower(): v for k, v in sorted(source_vocab.items())}
    fuzzy_overlap = exact_overlap

    for target_token, target_token_idx in sorted(target_vocab.items()):
        lowercase_target_token = target_token.lower()
        if fuzzy_overlap.get(target_token):
            continue
        if lowercase_source_vocab.get(lowercase_target_token):
            # same token but just different case found in source vocab
            fuzzy_overlap[target_token] = (
                target_token_idx,
                lowercase_source_vocab[lowercase_target_token],
            )
        elif fuzzy_whitespace and lowercase_source_vocab.get(
            " " + lowercase_target_token
        ):
            # same token with extra whitespace found in source vocab
            fuzzy_overlap[target_token] = (
                target_token_idx,
                lowercase_source_vocab[" " + lowercase_target_token],
            )
        elif fuzzy_whitespace and lowercase_source_vocab.get(
            lowercase_target_token.lstrip()
        ):
            # same token without extra whitespace found in source vocab
            fuzzy_overlap[target_token] = (
                target_token_idx,
                lowercase_source_vocab[lowercase_target_token.lstrip()],
            )
    return {
        target_tokenizer.convert_ids_to_tokens(v[0]): v
        for k, v in fuzzy_overlap.items()
    }


@torch.no_grad()
def focus_additional_token_initialization(
    fasttext_model,
    shared_tokens,
    new_tokens,
    target_embeddings: Tensor,
    p=1.0,
    temperature=1,
):
    def sanitized_fasttext_vector(token, fasttext_model):
        """
        Some tokens are not part of fasttext model even though they are in the target tokenizer vocab.
        Calling fasttext_model[<token>] will return combination of subword ngrams for OOV <token>.
        However, when the OOV token is short (e.g. 1 letter), there might be none and a zero-vector will be returned.
        This is bad, because a zero-vector leads to NAN in cosine similarity (division by zero).
        """
        ftv = fasttext_model[token]
        if sum(ftv) == 0:
            ftv = np.random.randn(*ftv.shape)
        return ftv

    new_token_fasttext_embs = OrderedDict(
        (
            (token, sanitized_fasttext_vector(token, fasttext_model))
            for token in new_tokens.keys()
        )
    )
    shared_token_fasttext_embs = OrderedDict(
        (
            (token, sanitized_fasttext_vector(token, fasttext_model))
            for token in shared_tokens.keys()
        )
    )

    new_token_ft_emb_matrix = np.asarray(
        [t for t in list(new_token_fasttext_embs.values())], dtype="float32"
    )
    shared_token_ft_emb_matrix = np.asarray(
        [t for t in list(shared_token_fasttext_embs.values())], dtype="float32"
    )

    from fastdist import fastdist

    new_to_shared_cosine_sims = fastdist.cosine_matrix_to_matrix(
        new_token_ft_emb_matrix, shared_token_ft_emb_matrix
    )

    shared_token_idx_to_target_vocab = list(shared_token_fasttext_embs.keys())
    for new_token, shared_token_cosine_sims in tqdm(
        zip(list(new_token_fasttext_embs.keys()), new_to_shared_cosine_sims),
        desc="FOCUS initialization...",
        total=len(new_to_shared_cosine_sims),
    ):
        ranked_shared_token_idxs = np.argsort(shared_token_cosine_sims)[::-1]
        ranked_shared_token_embs = np.sort(shared_token_cosine_sims)[::-1]

        import entmax

        sparsemax = entmax.sparsemax(
            torch.from_numpy(ranked_shared_token_embs.copy()) / temperature
        ).numpy()

        accumulated_prob_mass = 0.0
        convex_combination = torch.zeros_like(target_embeddings[0])
        for sparsemax_prob_mass, ranked_shared_token_idx in zip(
            sparsemax, ranked_shared_token_idxs
        ):
            if sparsemax_prob_mass == 0.0 or accumulated_prob_mass >= p:
                break
            ranked_shared_token_idx_in_target_vocab = shared_tokens[
                shared_token_idx_to_target_vocab[ranked_shared_token_idx]
            ]
            convex_combination += (
                sparsemax_prob_mass
                * target_embeddings[ranked_shared_token_idx_in_target_vocab]
            )
            accumulated_prob_mass += sparsemax_prob_mass

        # scale all coefficients s.t. we have convex combination (sum of all coefficients is 1 and each coeffcient is > 0)
        # post-hoc here because it's easier to implement
        # in case of p threshold == 1, this is a no-op
        if p < 1.0:
            convex_combination = convex_combination / accumulated_prob_mass

        # Initialize the new token embedding with the FOCUS combination
        target_embedding_idx = new_tokens[new_token]
        target_embeddings[target_embedding_idx] = convex_combination

    return target_embeddings.detach()


def FOCUS(
    target_tokenizer: PreTrainedTokenizer,
    source_tokenizer: PreTrainedTokenizer,
    source_embeddings: Tensor,
    target_training_data_path: str | None = None,
    fasttext_model_path: str | None = None,
    language_identifier: str | None = None,
    fasttext_epochs: int = 3,
    fasttext_embedding_dim: int = 300,
    processes: int | None = None,
    debug: bool = False,
) -> Tensor:
    """FOCUS initialization to create a well-initialized embedding matrix for a target tokenizer given pretrained embeddings.

    FOCUS uses an **auxiliary** fasttext embedding to initialize the target tokenizer's embedding matrix. You can either provide `target_training_data_path` to train a fasttext model on the given data, directly supply a pretrained fasttext model with `fasttext_model_path`.
    We also implement a method to use pretrained **word** embeddings with `language_identifier` (e.g. "de" for German) which is based on WECHSEL (Minixhofer et al., https://github.com/CPJKU/wechsel/).

    Args:
        target_tokenizer (PreTrainedTokenizer): The tokenizer for the target language.
        source_tokenizer (PreTrainedTokenizer): The tokenizer used for the source model.
        source_embeddings (Tensor): The embedding matrix of the pretrained source model.

        Choose one of the following three:
        - target_training_data_path (str | None, optional): Path to a text file containing lines of text in the target language. Used to train a fasttext embedding for FOCUS. Defaults to None.
        - fasttext_model_path (str | None, optional): Path to a pretrained fasttext model for the `target_tokenizer`'s tokens. Defaults to None.
        - language_identifier (str | None, optional): Two-letter language identifier. We will then download pretrained fasttext *word* embeddings for the target language and embed the tokens of the target tokenizer into the pretrained fasttext word embedding space. This is akin to the method used in the WECHSEL paper and different from what we used in the FOCUS paper. Defaults to None.

        If you provide `target_training_data_path` or `fasttext_model_path`, you can also provide the following two arguments:
        - epochs (int, optional): The number of epochs used to train the fasttext model, if necessary. Defaults to 3.
        - dim (int, optional): The embedding dimension for the fasttext model, if necessary. Defaults to 300.

        You can set `processes` (int, optional) to control the number of parallel processes used for training the fasttext model. If `None`, default to `multiprocessing.cpu_count()`.


    Returns:
        Tensor: The embedding matrix for the given target tokenizer, initialized with `FOCUS`.
    """
    fasttext_model = load_target_token_embedding(
        target_tokenizer=target_tokenizer,
        target_training_data_path=target_training_data_path,
        language_identifier=language_identifier,
        fasttext_model_path=fasttext_model_path,
        fasttext_epochs=fasttext_epochs,
        fasttext_embedding_dim=fasttext_embedding_dim,
        processes=processes,
    )

    target_token_set = set(target_tokenizer.get_vocab().keys())

    if isinstance(fasttext_model, dict):
        missing_tokens = target_token_set.difference(set(fasttext_model.keys()))
    else:
        missing_tokens = target_token_set.difference(set(fasttext_model.words))

    if debug and len(missing_tokens) > 0:
        logger.warning(
            f"{len(missing_tokens)} target tokens not in fasttext model: {missing_tokens}.  Note: a small number is okay."
        )

    overlapping_token_mapping = get_overlapping_tokens(
        target_tokenizer, source_tokenizer, fuzzy_search=True
    )

    target_embeddings = torch.zeros((len(target_tokenizer), source_embeddings.shape[1]))

    # Copy embeddings for overlapping tokens
    overlapping_tokens = {}
    for overlapping_token, (
        target_vocab_idx,
        source_vocab_idx,
    ) in overlapping_token_mapping.items():
        overlapping_tokens[overlapping_token] = target_vocab_idx
        target_embeddings[target_vocab_idx] = source_embeddings[source_vocab_idx]

    additional_tokens = {
        token: idx
        for token, idx in target_tokenizer.get_vocab().items()
        if not overlapping_tokens.get(token)
    }

    target_embeddings = focus_additional_token_initialization(
        fasttext_model, overlapping_tokens, additional_tokens, target_embeddings
    )

    return target_embeddings


def EXTEND(
    target_tokenizer: PreTrainedTokenizer,
    source_tokenizer: PreTrainedTokenizer,
    extended_tokenizer: PreTrainedTokenizer,
    source_embeddings: Tensor,
    target_training_data_path: str | None = None,
    fasttext_model_path: str | None = None,
    language_identifier: str | None = None,
    fasttext_epochs: int = 3,
    fasttext_embedding_dim: int = 300,
    processes: int | None = None,
):
    """
    Similar to FOCUS, but for vocabulary extension instead of replacement.
    `target_tokenizer` is the tokenizer used to extend the `source_tokenizer`'s vocabulary. The resulting tokenizer should be passed as `extended_tokenizer`.
    For other options, refer to the documentation of `FOCUS`.
    """
    fasttext_model = load_target_token_embedding(
        target_tokenizer=target_tokenizer,
        target_training_data_path=target_training_data_path,
        language_identifier=language_identifier,
        fasttext_model_path=fasttext_model_path,
        fasttext_epochs=fasttext_epochs,
        fasttext_embedding_dim=fasttext_embedding_dim,
        processes=processes,
    )

    overlapping_token_mapping = get_overlapping_tokens(
        extended_tokenizer, source_tokenizer, fuzzy_search=True
    )

    target_embeddings = torch.zeros(
        (len(extended_tokenizer), source_embeddings.shape[1])
    )

    # Copy embeddings for all overlapping tokens
    overlapping_tokens = {}
    for overlapping_token, (
        target_vocab_idx,
        source_vocab_idx,
    ) in overlapping_token_mapping.items():
        # Filter out all overlapping tokens that are not in the language-specific target vocab
        if target_tokenizer.get_vocab().get(overlapping_token) is not None:
            overlapping_tokens[overlapping_token] = target_vocab_idx
        target_embeddings[target_vocab_idx] = source_embeddings[source_vocab_idx]

    additional_tokens = {
        token: idx
        for token, idx in target_tokenizer.get_vocab().items()
        if not overlapping_tokens.get(token)
    }

    target_embeddings = focus_additional_token_initialization(
        fasttext_model, overlapping_tokens, additional_tokens, target_embeddings
    )

    return target_embeddings
