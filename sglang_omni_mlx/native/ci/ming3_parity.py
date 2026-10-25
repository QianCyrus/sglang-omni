"""Compare Ming frontend and cached decoder tensors with Python MLX Audio."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_audio.stt import load
from mlx_audio.stt.models.ming_v3_asr.ming_v3_asr import ASR_INSTRUCTION, Model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-bin", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    arguments = parser.parse_args()
    model: Model = load(str(arguments.model_path), strict=True)
    features = model.prepare_audio(str(arguments.audio))
    expected_audio = model.encode_audio(features)
    mx.eval(expected_audio)
    with tempfile.TemporaryDirectory(prefix="ming-parity-") as directory:
        root = Path(directory)
        subprocess.run(
            [
                str(arguments.runtime_bin / "ming3_asr_probe"),
                "encode",
                str(arguments.audio),
                str(root / "audio.safetensors"),
                str(arguments.model_path),
            ],
            check=True,
        )
        actual_audio = mx.load(str(root / "audio.safetensors"))
        feature_difference = mx.abs(
            features.astype(mx.float32)
            - actual_audio["features"].astype(mx.bfloat16).astype(mx.float32)
        )
        audio_difference = mx.abs(
            expected_audio.astype(mx.float32) - actual_audio["audio"].astype(mx.float32)
        )
        print(
            json.dumps(
                {
                    "feature_max_abs": mx.max(feature_difference).item(),
                    "audio_max_abs": mx.max(audio_difference).item(),
                },
                indent=2,
            ),
            flush=True,
        )
        np.testing.assert_allclose(
            np.array(features.astype(mx.float32)),
            np.array(actual_audio["features"].astype(mx.bfloat16).astype(mx.float32)),
            atol=0.0,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            np.array(expected_audio.astype(mx.float32)),
            np.array(actual_audio["audio"].astype(mx.float32)),
            atol=0.0,
            rtol=0.0,
        )
        ids = model.tokenizer.encode(
            ASR_INSTRUCTION + model.audio_tokens[0], add_special_tokens=False
        )
        ids += [model.config.audio_token_id] * expected_audio.shape[1]
        ids += model.tokenizer.encode(
            model.audio_tokens[1] + "English\t", add_special_tokens=False
        )
        _, embeddings = model.inject_audio(ids, expected_audio)
        mx.save_safetensors(str(root / "input.safetensors"), {"embeddings": embeddings})
        for chunk_size in (embeddings.shape[1], 31, 1):
            cache = model.make_cache()
            outputs = []
            for start in range(0, embeddings.shape[1], chunk_size):
                hidden = model.model(
                    inputs_embeds=embeddings[:, start : start + chunk_size], cache=cache
                )
                mx.eval(hidden, [state.state for state in cache])
                outputs.append(hidden)
            expected = {
                "hidden": mx.concatenate(outputs, axis=1),
                "logits": model.lm_head(outputs[-1][:, -1:]).astype(mx.float32),
            }
            for layer, state in enumerate(cache):
                if model.config.text_config.uses_mla(layer):
                    expected[f"{layer}.keys"], expected[f"{layer}.values"] = state.state
                else:
                    expected[f"{layer}.recurrent"] = state[3]
                    expected.update(
                        {f"{layer}.conv.{index}": state[index] for index in range(3)}
                    )
            mx.eval(list(expected.values()))
            subprocess.run(
                [
                    str(arguments.runtime_bin / "ming3_asr_probe"),
                    "forward",
                    str(root / "input.safetensors"),
                    str(root / "forward.safetensors"),
                    str(arguments.model_path),
                    str(chunk_size),
                ],
                check=True,
            )
            actual = mx.load(str(root / "forward.safetensors"))
            differences = {
                name: mx.max(
                    mx.abs(value.astype(mx.float32) - actual[name].astype(mx.float32))
                ).item()
                for name, value in expected.items()
            }
            print(
                json.dumps(
                    {"chunk_size": chunk_size, "max_abs": differences}, indent=2
                ),
                flush=True,
            )
            for name, value in expected.items():
                np.testing.assert_allclose(
                    np.array(value.astype(mx.float32)),
                    np.array(actual[name].astype(mx.float32)),
                    atol=0.0,
                    rtol=0.0,
                    err_msg=f"chunk={chunk_size}, tensor={name}",
                )


if __name__ == "__main__":
    main()
