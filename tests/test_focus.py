import random

import numpy as np
import pytest
import torch
from pytest_mock import MockerFixture

from src.deepfocus import FOCUS
from src.deepfocus.focus import get_overlapping_tokens

"""
NOTE: We use the fasttext-wordlevel option here so that we do not need to host a pretrained tokenlevel fasttext model somewhere for testing with GitHub Actions.
"""


@pytest.mark.reproducibility
def test_focus_reproducibility(mocker: MockerFixture, de_tokenizer, xlmr_tokenizer, xlmr_embeddings):
    target_tokenizer = de_tokenizer
    source_tokenizer = xlmr_tokenizer
    source_embeddings = xlmr_embeddings

    target_embeddings1 = FOCUS(
        source_embeddings=source_embeddings,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        auxiliary_embedding_mode="fasttext-wordlevel",
        language_identifier="de",
        exact_match_all=True,
        seed=1,
    )

    # Do some stuff that might influence the random state
    torch.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)
    torch.randn(100)

    target_embeddings2 = FOCUS(
        source_embeddings=source_embeddings,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        auxiliary_embedding_mode="fasttext-wordlevel",
        language_identifier="de",
        exact_match_all=True,
        seed=1,
    )

    assert torch.allclose(target_embeddings1, target_embeddings2)

    for i in range(len(target_embeddings1)):
        assert torch.equal(target_embeddings1[i], target_embeddings2[i])


@pytest.mark.main
def test_focus_with_xlmr(mocker: MockerFixture, de_tokenizer, xlmr_tokenizer, xlmr_embeddings):
    target_tokenizer = de_tokenizer
    source_tokenizer = xlmr_tokenizer
    source_embeddings = xlmr_embeddings

    target_embeddings = FOCUS(
        source_embeddings=xlmr_embeddings,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        language_identifier="de",
        auxiliary_embedding_mode="fasttext-wordlevel",
        exact_match_all=True,
    )
    ########################
    ###### Assertions ######
    ########################

    # assert that special tokens keep the same embedding
    assert torch.equal(
        target_embeddings[target_tokenizer.get_vocab()["<mask>"]],
        source_embeddings[source_tokenizer.get_vocab()["<mask>"]],
    )
    assert torch.equal(
        target_embeddings[target_tokenizer.get_vocab()["</s>"]],
        source_embeddings[source_tokenizer.get_vocab()["</s>"]],
    )
    assert torch.equal(
        target_embeddings[target_tokenizer.get_vocab()["<pad>"]],
        source_embeddings[source_tokenizer.get_vocab()["<pad>"]],
    )
    assert torch.equal(
        target_embeddings[target_tokenizer.get_vocab()["<s>"]],
        source_embeddings[source_tokenizer.get_vocab()["<s>"]],
    )

    # assert that no row in the target embedding matrix is all zeros
    assert torch.abs(target_embeddings).sum(dim=1).min() > 0

    overlap, additional = get_overlapping_tokens(
        target_tokenizer,
        source_tokenizer,
        exact_match_all=True,
        match_symbols=False,
        fuzzy_match_all=False,
    )

    # overlapping tokens should be copied
    for token, token_info in overlap.items():
        assert torch.equal(
            target_embeddings[token_info.target.id],
            source_embeddings[token_info.source[0].id],
        )


@pytest.mark.extend
def test_extend_focus_with_xlmr(
    de_extended_tokenizer,
    de_extend_tokenizer,
    xlmr_tokenizer,
    xlmr_embeddings,
):
    target_tokenizer = de_extended_tokenizer
    source_tokenizer = xlmr_tokenizer
    source_embeddings = xlmr_embeddings

    target_embeddings = FOCUS(
        source_embeddings=xlmr_embeddings,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        language_identifier="de",
        auxiliary_embedding_mode="fasttext-wordlevel",
        exact_match_all=True,
        extend_tokenizer=de_extend_tokenizer,
    )
    ########################
    ###### Assertions ######
    ########################

    # assert that special tokens keep the same embedding
    assert torch.equal(
        target_embeddings[target_tokenizer.get_vocab()["<mask>"]],
        source_embeddings[source_tokenizer.get_vocab()["<mask>"]],
    )
    assert torch.equal(
        target_embeddings[target_tokenizer.get_vocab()["</s>"]],
        source_embeddings[source_tokenizer.get_vocab()["</s>"]],
    )
    assert torch.equal(
        target_embeddings[target_tokenizer.get_vocab()["<pad>"]],
        source_embeddings[source_tokenizer.get_vocab()["<pad>"]],
    )
    assert torch.equal(
        target_embeddings[target_tokenizer.get_vocab()["<s>"]],
        source_embeddings[source_tokenizer.get_vocab()["<s>"]],
    )

    # assert that no row in the target embedding matrix is all zeros
    assert torch.abs(target_embeddings).sum(dim=1).min() > 0

    overlap, additional = get_overlapping_tokens(
        target_tokenizer,
        source_tokenizer,
        exact_match_all=True,
        match_symbols=False,
        fuzzy_match_all=False,
    )

    # overlapping tokens should be copied
    for token, token_info in overlap.items():
        assert torch.equal(
            target_embeddings[token_info.target.id],
            source_embeddings[token_info.source[0].id],
        )

    source_vocab = source_tokenizer.get_vocab()
    target_vocab = target_tokenizer.get_vocab()

    # manual sanity check: all tokens in the source tokenizer should have a corresponding embedding in the target embedding
    for token in source_vocab.keys():
        source_id = source_vocab[token]
        target_id = target_vocab[token]
        assert torch.equal(
            target_embeddings[target_id],
            source_embeddings[source_id],
        )
