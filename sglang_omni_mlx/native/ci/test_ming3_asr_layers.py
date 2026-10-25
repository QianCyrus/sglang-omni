"""Numerical recurrence and tokenizer contracts used by Ming native inference."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file
from server_fixture import DATA_ROOT, RUNTIME_BIN
from tokenizers import Tokenizer

pytestmark = pytest.mark.skipif(
    not RUNTIME_BIN or not DATA_ROOT, reason="set NATIVE_RUNTIME_BIN and CI_DATA_ROOT"
)


def test_ming_delta_matches_recurrence_and_chunked_state(tmp_path: Path) -> None:
    generator = np.random.default_rng(42)
    shape = (2, 9, 2, 128)
    queries = generator.normal(size=shape).astype(np.float32)
    keys = generator.normal(size=shape).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=-1, keepdims=True) * np.sqrt(128)
    keys /= np.linalg.norm(keys, axis=-1, keepdims=True)
    tensors = {
        "queries": queries,
        "keys": keys,
        "values": generator.normal(size=shape).astype(np.float32),
        "decay": generator.uniform(0.1, 0.95, size=shape).astype(np.float32),
        "beta": generator.uniform(size=shape[:-1]).astype(np.float32),
        "state": generator.normal(size=(2, 2, 128, 128)).astype(np.float32),
    }
    state = tensors["state"].copy()
    expected = []
    for token in range(shape[1]):
        state *= tensors["decay"][:, token, :, None, :]
        key = keys[:, token, :, None, :]
        prediction = (state * key).sum(-1)
        delta = (tensors["values"][:, token] - prediction) * tensors["beta"][
            :, token, :, None
        ]
        state += delta[..., None] * key
        expected.append((state * queries[:, token, :, None, :]).sum(-1))
    save_file(tensors, tmp_path / "input.safetensors")
    subprocess.run(
        [
            str(Path(RUNTIME_BIN) / "ming3_asr_probe"),
            "delta",
            str(tmp_path / "input.safetensors"),
            str(tmp_path / "output.safetensors"),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    actual = load_file(tmp_path / "output.safetensors")
    np.testing.assert_allclose(
        actual["output"], np.stack(expected, axis=1), atol=2e-6, rtol=2e-5
    )
    np.testing.assert_allclose(actual["state"], state, atol=2e-6, rtol=2e-5)
    carried = tensors["state"]
    outputs = []
    for start, end in ((0, 4), (4, 9)):
        chunk = {
            name: np.ascontiguousarray(value[:, start:end])
            for name, value in tensors.items()
            if name != "state"
        }
        chunk["state"] = carried
        save_file(chunk, tmp_path / "chunk.safetensors")
        subprocess.run(
            [
                str(Path(RUNTIME_BIN) / "ming3_asr_probe"),
                "delta",
                str(tmp_path / "chunk.safetensors"),
                str(tmp_path / "chunk-output.safetensors"),
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )
        result = load_file(tmp_path / "chunk-output.safetensors")
        carried = result["state"]
        outputs.append(result["output"])
    np.testing.assert_array_equal(np.concatenate(outputs, axis=1), actual["output"])
    np.testing.assert_array_equal(carried, actual["state"])


def test_ming_delta_rejects_head_width_not_supported_by_kernel(
    tmp_path: Path,
) -> None:
    shape = (1, 1, 1, 31)
    tensors = {
        "queries": np.zeros(shape, dtype=np.float32),
        "keys": np.zeros(shape, dtype=np.float32),
        "values": np.zeros(shape, dtype=np.float32),
        "decay": np.ones(shape, dtype=np.float32),
        "beta": np.ones(shape[:-1], dtype=np.float32),
        "state": np.zeros((1, 1, 31, 31), dtype=np.float32),
    }
    input_path = tmp_path / "input.safetensors"
    save_file(tensors, input_path)

    completed = subprocess.run(
        [
            str(Path(RUNTIME_BIN) / "ming3_asr_probe"),
            "delta",
            str(input_path),
            str(tmp_path / "output.safetensors"),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode != 0
    assert "head_width divisible by 32" in completed.stderr


@pytest.mark.parametrize("layout", ["tokenizer_json", "vocab_merges"])
def test_byte_level_tokenizer_matches_checkpoint(tmp_path: Path, layout: str) -> None:
    directory = Path(DATA_ROOT) / "models" / "mlx-community_Ming-3.0-asr-tiny-4bit"
    tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
    if layout == "vocab_merges":
        checkpoint = json.loads((directory / "tokenizer.json").read_text())
        (tmp_path / "vocab.json").write_text(json.dumps(checkpoint["model"]["vocab"]))
        (tmp_path / "merges.txt").write_text(
            "#version: 0.2\n"
            + "\n".join(" ".join(pair) for pair in checkpoint["model"]["merges"])
            + "\n"
        )
        (tmp_path / "tokenizer_config.json").write_bytes(
            (directory / "tokenizer_config.json").read_bytes()
        )
        checkpoint["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] = (
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}"
            r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
        )
        tokenizer = Tokenizer.from_str(json.dumps(checkpoint))
        directory = tmp_path
    else:
        pass
    texts = [
        "Please recognize the language of this speech and transcribe it, placing a `\t` between the identified language and the transcribe.<audio>",
        "English\t你好世界 1234567890",
        "I'm WE'RE they've isn't 42.5!\n\n\t  ",
        "Cafe\u0301 naïve 中文🙂\r\n!  \n",
        "<audio><audioPatch><audioPatch></audio><|role_end|>",
        "",
    ]
    (tmp_path / "texts.json").write_text(json.dumps(texts))
    subprocess.run(
        [
            str(Path(RUNTIME_BIN) / "ming3_asr_probe"),
            "tokenizer",
            str(tmp_path / "texts.json"),
            str(tmp_path / "tokens.json"),
            str(directory),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    actual = json.loads((tmp_path / "tokens.json").read_text())
    for text, result in zip(texts, actual, strict=True):
        expected = tokenizer.encode(text, add_special_tokens=False).ids
        assert result["ids"] == expected
        assert result["decoded"] == tokenizer.decode(
            expected, skip_special_tokens=False
        )
