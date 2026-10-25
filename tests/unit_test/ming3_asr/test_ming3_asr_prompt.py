# SPDX-License-Identifier: Apache-2.0
"""Transcription prompt layout and transcript extraction for Ming 3 ASR."""

from unittest.mock import Mock

import pytest
from transformers import PreTrainedTokenizerBase

from sglang_omni.models.ming3_asr.preprocessing import (
    ASR_INSTRUCTION,
    AUDIO_END_TOKEN,
    AUDIO_PATCH_TOKEN,
    AUDIO_START_TOKEN,
    Ming3ASRPreprocessor,
    build_transcription_prompt,
    extract_transcript,
    resolve_eos_token_ids,
)
from sglang_omni.modules.ming_audio.audio_encoder import AudioEncoderConfig
from sglang_omni.serve.openai_errors import generation_error_status_code

AUDIO_TOKEN_COUNT = 3
AUDIO_SPAN = AUDIO_START_TOKEN + AUDIO_PATCH_TOKEN * AUDIO_TOKEN_COUNT + AUDIO_END_TOKEN
MING3_AUDIO_CONFIG = AudioEncoderConfig(
    n_mels=128,
    n_audio_ctx=3000,
    n_audio_state=1280,
    n_audio_head=20,
    n_audio_layer=32,
    downsampler_kernel_size_list=[3],
    downsampler_stride_list=[2],
)


def test_prompt_is_instruction_followed_by_audio_span() -> None:
    assert (
        build_transcription_prompt(AUDIO_TOKEN_COUNT, None, None)
        == ASR_INSTRUCTION + AUDIO_SPAN
    )


@pytest.mark.parametrize(
    ("language", "answer_prefix"), [("en", "English\t"), ("zh", "Chinese\t")]
)
def test_known_language_prefills_language_label(
    language: str, answer_prefix: str
) -> None:
    assert build_transcription_prompt(AUDIO_TOKEN_COUNT, language, None) == (
        ASR_INSTRUCTION + AUDIO_SPAN + answer_prefix
    )


def test_context_follows_instruction_before_audio() -> None:
    context = "This is an audio about films."
    assert (
        build_transcription_prompt(AUDIO_TOKEN_COUNT, None, context)
        == ASR_INSTRUCTION + " " + context + AUDIO_SPAN
    )


@pytest.mark.parametrize(
    ("vocabulary", "tokenizer_token_id", "error_message"),
    [
        ({}, 157168, "missing the <audioPatch> token"),
        ({AUDIO_PATCH_TOKEN: 157167}, 157167, "audio token ID mismatch"),
    ],
)
def test_preprocessor_rejects_invalid_audio_token(
    vocabulary: dict[str, int], tokenizer_token_id: int, error_message: str
) -> None:
    tokenizer = Mock(spec=PreTrainedTokenizerBase)
    tokenizer.get_vocab.return_value = vocabulary
    tokenizer.convert_tokens_to_ids.return_value = tokenizer_token_id

    with pytest.raises(ValueError, match=error_message):
        Ming3ASRPreprocessor(tokenizer, MING3_AUDIO_CONFIG, 64, 157168)


def test_preprocessor_accepts_audio_token_matching_checkpoint() -> None:
    tokenizer = Mock(spec=PreTrainedTokenizerBase)
    tokenizer.get_vocab.return_value = {AUDIO_PATCH_TOKEN: 157168}
    tokenizer.convert_tokens_to_ids.return_value = 157168

    preprocessor = Ming3ASRPreprocessor(tokenizer, MING3_AUDIO_CONFIG, 64, 157168)

    assert preprocessor.audio_token_id == 157168


def test_resolve_eos_token_ids_includes_checkpoint_and_tokenizer_ids() -> None:
    tokenizer = Mock(spec=PreTrainedTokenizerBase)
    tokenizer.eos_token_id = 156895

    assert resolve_eos_token_ids(156892, tokenizer) == {156892, 156895}


@pytest.mark.parametrize(
    ("language", "context"), [("fr", None), (None, "first line\nsecond line")]
)
def test_unsupported_requests_are_rejected_as_caller_errors(
    language: str | None, context: str | None
) -> None:
    with pytest.raises(ValueError) as rejection:
        build_transcription_prompt(AUDIO_TOKEN_COUNT, language, context)
    assert generation_error_status_code(rejection.value) == 400


@pytest.mark.parametrize(
    ("generated_text", "transcript"),
    [
        ("English\thello there ", "hello there"),
        ("English hello there", "hello there"),
        ("hello there", "hello there"),
    ],
)
def test_transcript_drops_language_label(generated_text: str, transcript: str) -> None:
    assert extract_transcript(generated_text) == transcript
