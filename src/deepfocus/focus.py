from typing import Literal

import entmax
import numpy as np
import torch
from fastdist import fastdist
from torch import Tensor
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizer

from .fasttext_embs import load_target_token_embedding
from .logger import logger
from .vocab_helper import NewToken, OverlappingToken, canonicalize_vocab, construct_vocab_view, is_numerical_symbol_etc


def get_overlapping_tokens(
    target_tokenizer: PreTrainedTokenizer,
    source_tokenizer: PreTrainedTokenizer,
    match_symbols: bool,
    exact_match_all: bool,
    fuzzy_match_all: bool,
):
    """Returns overlapping tokens between two tokenizers. There are several options to select which tokens count as overlapping tokens.

    Args:
        target_tokenizer (PreTrainedTokenizer): The target tokenizer.
        source_tokenizer (PreTrainedTokenizer): The source tokenizer.
        match_symbols (bool): Tokens that satisfy `token.isnumeric() or all(c in string.punctuation for c in token) or token.isspace()` are considered.
        exact_match_all (bool): All tokens that match exactly are considered.
        fuzzy_match_all (bool): All tokens that match ignoring whitespace and case are considered.

    Returns:
        `(dict[str, OverlappingToken], dict[str, NewToken])`: A tuple with (1) information about overlapping tokens and (2) additional tokens in the target tokenizer.
    """
    target_vocab = target_tokenizer.get_vocab()
    source_vocab = source_tokenizer.get_vocab()

    canonical_source_vocab = canonicalize_vocab(source_vocab, source_tokenizer, "source")
    canonical_target_vocab = canonicalize_vocab(target_vocab, target_tokenizer, "target")

    overlap: dict[str, OverlappingToken] = {}
    additional_tokens: dict[str, NewToken] = {}
    exact_src_vocab = construct_vocab_view(canonical_source_vocab, "canonical_form")
    fuzzy_src_vocab = construct_vocab_view(canonical_source_vocab, "fuzzy_form")

    for _, target_token_info in tqdm(
        canonical_target_vocab.items(),
        desc="Getting overlapping tokens...",
        leave=False,
    ):
        # Exact match for symbols
        if (
            match_symbols
            and is_numerical_symbol_etc(target_token_info.fuzzy_form, target_tokenizer)
            and (exact_src_vocab.get(target_token_info.canonical_form) or fuzzy_src_vocab.get(target_token_info.fuzzy_form))
        ):
            overlap[target_token_info.native_form] = OverlappingToken(
                target=target_token_info,
                source=(
                    exact_src_vocab.get(target_token_info.canonical_form) or fuzzy_src_vocab.get(target_token_info.fuzzy_form)
                ),
                descriptor="numerical_symbol",
            )
        # General exact match
        elif exact_match_all and exact_src_vocab.get(target_token_info.canonical_form):
            overlap[target_token_info.native_form] = OverlappingToken(
                target=target_token_info,
                source=exact_src_vocab[target_token_info.canonical_form],
                descriptor="exact_match",
            )
        # General fuzzy match
        elif fuzzy_match_all and fuzzy_src_vocab.get(target_token_info.fuzzy_form):
            overlap[target_token_info.native_form] = OverlappingToken(
                target=target_token_info,
                source=fuzzy_src_vocab[target_token_info.fuzzy_form],
                descriptor="fuzzy_match",
            )
        # No match - it's a NewToken
        else:
            additional_tokens[target_token_info.native_form] = NewToken(target=target_token_info)
    return overlap, additional_tokens


def is_very_rare_token(token, fasttext_model):
    """
    We want to filter out some "bad" tokens.
    These are tokens that are so rare that they did not get an embedding in the fasttext model.
    If using pretrained word embeddings, these are tokens where no subwords are part of the pretrained word fasttext model.
    These tokens will be initialized with a random embedding.
    """
    return token not in fasttext_model or np.absolute(fasttext_model[token]).sum() == 0


@torch.no_grad()
def FOCUS(
    target_tokenizer: PreTrainedTokenizer,
    source_tokenizer: PreTrainedTokenizer,
    source_embeddings: Tensor,
    # Auxiliary embedding args
    auxiliary_embedding_mode: Literal["fasttext-tokenlevel", "fasttext-wordlevel"] = "fasttext-tokenlevel",
    target_training_data_path: str | None = None,
    fasttext_model_path: str | None = None,
    language_identifier: str | None = None,
    fasttext_model_epochs: int = 3,
    fasttext_model_dim: int = 100,
    fasttext_model_min_count: int = 10,
    # match options
    exact_match_all: bool = True,
    match_symbols: bool = False,
    fuzzy_match_all: bool = False,
    extend_tokenizer: PreTrainedTokenizer | None = None,
    processes: int | None = None,
    seed: int = 42,
    device="cpu",
    verbosity: Literal["debug", "info", "silent"] = "info",
):
    """FOCUS method for transferring pretrained token embeddings to a different language from Dobler and de Melo (2023).

    Args:
        target_tokenizer (PreTrainedTokenizer): The new tokenizer in the target language.
        source_tokenizer (PreTrainedTokenizer): The tokenizer for the pretrained source embeddings.
        source_embeddings (Tensor): The pretrained source embeddings tensor.
        auxiliary_embedding_mode ("fasttext-tokenlevel" or "fasttext-wordlevel"): The type of auxiliary embeddings to use. Defaults to "fasttext-tokenlevel".
        target_training_data_path (str | None, optional): Path to a file containing lines of text in the target language for training a fasttext model. Only necessary if using `fasttext-tokenlevel`. Defaults to None.
        fasttext_model_path (str | None, optional): Path to a pretrained fasttext model for the target tokenizer. Defaults to None.
        language_identifier (str | None, optional): Two-letter language code for downloading pretrained fasttext word embeddings if using `fasttext-wordlevel`. Defaults to None.
        fasttext_model_epochs (int, optional): Number of epochs if training a custom fasttext model. Defaults to 3.
        fasttext_model_dim (int, optional): Dimension size if training a custom fasttext model. Defaults to 100.
        fasttext_model_min_count (int, optional): Minimum number of occurrences for a token to be included if training a custom fasttext model. Defaults to 10.
        exact_match_all (bool, optional): Match all overlapping tokens if they are an exact match. Defaults to False.
        match_symbols (bool, optional): Match overlapping symbolic tokens. Defaults to False.
        fuzzy_match_all (bool, optional): Match all overlapping tokens with fuzzy matching (whitespace and case). Defaults to False.
        extend_tokenizer (PreTrainedTokenizer | None, optional): If extending a tokenizer instead of vocabulary replacement, this should be the tokenizer that was used to extend the `source_tokenizer` (i.e. a target language specific tokenizer). The `target_tokenizer` should be the *extended* tokenizer. Defaults to None.
        processes (int | None, optional): Number of processes for parallelized workloads. Defaults to None, which uses heuristics based on available hardware.
        seed (int, optional): Defaults to 42.
        device (str | torch.device, optional): Defaults to "cpu".
        verbosity ("debug", "info", "silent", optional): Defaults to "info".

    Returns:
        Tensor: A tensor of shape `(len(target_tokenizer), embedding_dim)` with the initialized embeddings.
    """
    mode = {"debug": "dev", "info": "package", "silent": "silent"}[verbosity]
    logger.config(mode=mode)
    logger.info(f"Starting FOCUS initialization for target vocabulary with {len(target_tokenizer)} tokens...")
    ###########################################################
    # 1. Load auxiliary embedding model for target vocabulary #
    ###########################################################
    if auxiliary_embedding_mode == "fasttext-tokenlevel":
        if not target_training_data_path and not fasttext_model_path:
            raise ValueError(
                "You need to provide a path to training data or pretrained fasttext model for fasttext-tokenlevel auxiliary embeddings."
            )
        fasttext_model = load_target_token_embedding(
            target_tokenizer=extend_tokenizer or target_tokenizer,
            target_training_data_path=target_training_data_path,
            fasttext_model_path=fasttext_model_path,
            epochs=fasttext_model_epochs,
            dim=fasttext_model_dim,
            min_count=fasttext_model_min_count,
            processes=processes,
        )
    elif auxiliary_embedding_mode == "fasttext-wordlevel":
        if not language_identifier:
            raise ValueError(
                "You need to provide a language identifier (e.g. de for German) for fasttext-wordlevel auxiliary embeddings."
            )
        fasttext_model = load_target_token_embedding(
            target_tokenizer=extend_tokenizer or target_tokenizer,
            language_identifier=language_identifier,
            processes=processes,
        )

    #################################################################
    # 2. Get overlapping tokens between source and target tokenizer #
    #################################################################
    overlapping_tokens, new_tokens = get_overlapping_tokens(
        target_tokenizer=target_tokenizer,
        source_tokenizer=source_tokenizer,
        match_symbols=match_symbols,
        exact_match_all=exact_match_all,
        fuzzy_match_all=fuzzy_match_all,
    )

    # Sort to ensure same order every time (especially important when executing on multiple ranks)
    sorted_overlapping_tokens = sorted(overlapping_tokens.items(), key=lambda x: x[1].target.id)
    sorted_new_tokens = sorted(new_tokens.items(), key=lambda x: x[1].target.id)
    logger.debug(f"Found {len(sorted_overlapping_tokens)} overlapping tokens.")

    ##########################################################
    # 3. Clean overlap + get auxiliary embeddings for tokens #
    ##########################################################
    # Clean overlapping tokens
    extend_tokenizer_vocab = extend_tokenizer.get_vocab() if extend_tokenizer else None
    very_rare_overlapping_tokens = []

    for token, overlapping_token_info in tqdm(
        sorted_overlapping_tokens,
        desc="Populating auxiliary embeddings for overlapping token...",
        leave=False,
    ):
        embs_lst = [source_embeddings[s.id] for s in overlapping_token_info.source]
        overlapping_tokens[token].source_embedding = embs_lst[0]

        if len(embs_lst) > 1:
            logger.warning(
                f"{token} has multiple source embeddings (using first): {[s.native_form for s in overlapping_token_info.source][:min(5, len(embs_lst))]}"
            )

        # Filter some tokens so that they are not used for FOCUS
        if extend_tokenizer and not extend_tokenizer_vocab.get(overlapping_token_info.target.native_form):
            # if extending, we do not want to use tokens that are not in the language-specific tokenizer
            overlapping_tokens[token].use_for_focus = False
        elif is_very_rare_token(token, fasttext_model):
            very_rare_overlapping_tokens.append(token)
            overlapping_tokens[token].use_for_focus = False
        else:
            overlapping_tokens[token].auxiliary_embedding = fasttext_model[token]

    logger.debug(
        f"Pruned {len(very_rare_overlapping_tokens)} overlapping tokens because they do not have an auxiliary embedding: {very_rare_overlapping_tokens}"
    )

    # Clean new tokens, mark "bad" tokens for random init
    random_init_new_tokens: list[NewToken] = []
    for token, new_token_info in tqdm(
        sorted_new_tokens,
        desc="Populating auxiliary embeddings for non-overlapping token...",
        leave=False,
    ):
        if is_very_rare_token(new_token_info.target.native_form, fasttext_model):
            random_init_new_tokens.append(new_token_info)
            del new_tokens[token]
        else:
            new_token_info.auxiliary_embedding = fasttext_model[token]

    logger.debug(f"Will initialize {len(random_init_new_tokens)} new tokens randomly.")
    logger.debug(f"{[t.target.native_form for t in random_init_new_tokens]}", escape=True)

    ####################################################
    # 4. Copy source embeddings for overlapping tokens #
    ####################################################
    target_embeddings = torch.zeros((len(target_tokenizer), source_embeddings.shape[1]), device=device)
    for _, overlapping_token in sorted_overlapping_tokens:
        target_embeddings[overlapping_token.target.id] = overlapping_token.source_embedding
    logger.success(f"Copied embeddings for {len(overlapping_tokens)} overlapping tokens.")

    ###########################################################
    # 5. Initialize "bad" new tokens from normal distribution #
    ###########################################################
    emb_mean = source_embeddings.mean(dim=0)
    emb_std = source_embeddings.std(dim=0)
    gen = torch.Generator(device=device).manual_seed(seed)
    for ood_new_token in random_init_new_tokens:
        target_embeddings[ood_new_token.target.id] = torch.normal(emb_mean, emb_std, generator=gen)
    logger.info(
        f"Initialized {len(random_init_new_tokens)} new tokens from N(source_mean, source_std) because they do not have auxiliary embeddings (this is okay if it's not too many)."
    )

    #######################################################
    # 6. Finally, initialize additional tokens with FOCUS #
    #######################################################
    overlapping_tokens_for_focus = {k: v for k, v in sorted_overlapping_tokens if v.use_for_focus}
    target_embeddings = focus_additional_token_initialization(
        overlapping_tokens_for_focus, new_tokens, target_embeddings, device=device
    )
    logger.success(f"🎯 Initialized {len(new_tokens)} new tokens with FOCUS 🎯")
    return target_embeddings.detach()


def focus_additional_token_initialization(
    overlapping_tokens: dict[str, OverlappingToken],
    new_tokens: dict[str, NewToken],
    target_embeddings: Tensor,
    device: torch.device | str | None = None,
):
    # Convert to lists to ensure same order (`.values()` might not guarantee same order every time)
    new_tokens_lst = list(new_tokens.values())
    overlapping_tokens_lst = list(overlapping_tokens.values())

    # Convert to numpy arrays for fastdist
    new_auxiliary_embedding_matrix = np.asarray([t.auxiliary_embedding.tolist() for t in new_tokens_lst], dtype="float32")
    overlapping_auxiliary_embedding_matrix = np.asarray(
        [t.auxiliary_embedding.tolist() for t in overlapping_tokens_lst],
        dtype="float32",
    )

    logger.debug("Computing distance matrix...")
    similarity_matrix = fastdist.cosine_matrix_to_matrix(
        new_auxiliary_embedding_matrix,
        overlapping_auxiliary_embedding_matrix,
    )

    # Not needed anymore, save memory
    del new_auxiliary_embedding_matrix
    del overlapping_auxiliary_embedding_matrix

    logger.debug("Computing new embeddings...")

    # Do `torch.stack` once outside of loop to save time
    overlapping_src_embs = [t.source_embedding for t in overlapping_tokens_lst]
    overlapping_src_embs = torch.stack(overlapping_src_embs)

    for new_token_idx in tqdm(
        range(len(new_tokens_lst)),
        desc="FOCUS initialization...",
        total=len(new_tokens_lst),
    ):
        overlapping_emb_weights: Tensor = entmax.sparsemax(torch.from_numpy(similarity_matrix[new_token_idx]).to(device))

        # performance optimization
        mask = overlapping_emb_weights > 0.0
        masked_overlapping_emb_weights = overlapping_emb_weights[mask]
        masked_overlapping_src_embs = overlapping_src_embs[mask]

        weighted_src_embs = torch.mul(masked_overlapping_src_embs, masked_overlapping_emb_weights.unsqueeze(1))
        # It's a convex combination because the weights sum up to 1
        convex_combination = torch.sum(weighted_src_embs, dim=0)

        new_token_target_vocab_idx = new_tokens_lst[new_token_idx].target.id
        target_embeddings[new_token_target_vocab_idx] = convex_combination
    return target_embeddings
