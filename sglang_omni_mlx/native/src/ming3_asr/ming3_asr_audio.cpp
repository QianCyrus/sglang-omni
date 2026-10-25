#include "ming3_asr/ming3_asr_model.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>

#include "common/audio.h"
#include "qwen3_asr/qwen3_asr_transcriber.h"

namespace ming3_asr {

namespace mx = mlx::core;

void ValidateAudio(const std::vector<float> &samples) {
  if (samples.size() <= qwen3_asr::kFftSize / 2 ||
      !std::all_of(samples.begin(), samples.end(),
                   [](float sample) { return std::isfinite(sample); })) {
    throw std::invalid_argument("Ming ASR requires finite mono audio with more "
                                "than 200 samples at 16 kHz");
  } else {
  }
}

mx::array AudioFeatures(const std::vector<float> &samples) {
  ValidateAudio(samples);
  constexpr int fft_size = qwen3_asr::kFftSize;
  constexpr int mel_bins = qwen3_asr::kMelBinCount;
  static const auto window = [] {
    std::vector<float> values(fft_size);
    for (int sample = 0; sample < fft_size; ++sample) {
      values[sample] = static_cast<float>(
          0.5 * (1 - std::cos(2 * M_PI * sample / fft_size)));
    }
    return mx::array(values.data(), {fft_size});
  }();
  static const auto filters = [] {
    const double log_step = std::log(6.4) / 27.0;
    const double max_mel = 15.0 + std::log(8.0) / log_step;
    const auto mel_points = mx::linspace(0.0, max_mel, mel_bins + 2);
    const auto frequency_points = mx::where(
        mx::greater_equal(mel_points, mx::array(15.0f)),
        mx::multiply(
            mx::array(1000.0f),
            mx::exp(mx::multiply(mx::array(static_cast<float>(log_step)),
                                 mx::subtract(mel_points, mx::array(15.0f))))),
        mx::multiply(mx::array(static_cast<float>(200.0 / 3.0)), mel_points));
    const auto differences =
        mx::subtract(mx::slice(frequency_points, {1}, {mel_bins + 2}),
                     mx::slice(frequency_points, {0}, {mel_bins + 1}));
    const auto slopes = mx::subtract(
        mx::expand_dims(frequency_points, 0),
        mx::expand_dims(mx::linspace(0.0, 8000.0, fft_size / 2 + 1), 1));
    const auto down = mx::divide(
        mx::negative(mx::slice(slopes, {0, 0}, {fft_size / 2 + 1, mel_bins})),
        mx::slice(differences, {0}, {mel_bins}));
    const auto up =
        mx::divide(mx::slice(slopes, {0, 2}, {fft_size / 2 + 1, mel_bins + 2}),
                   mx::slice(differences, {1}, {mel_bins + 1}));
    const auto normalization = mx::divide(
        mx::array(2.0f),
        mx::subtract(mx::slice(frequency_points, {2}, {mel_bins + 2}),
                     mx::slice(frequency_points, {0}, {mel_bins})));
    auto result =
        mx::multiply(mx::maximum(mx::zeros_like(down), mx::minimum(down, up)),
                     normalization);
    mx::eval(result);
    return result;
  }();
  return mx::astype(
      mx::transpose(qwen3_asr::LogMel(
          samples, qwen3_asr::AudioLayout::kReference, window, filters, false)),
      mx::bfloat16);
}

mx::array MingModel::EncodeChunk(const mx::array &features,
                                 const std::atomic<bool> &cancel) const {
  auto hidden = qwen3_asr::Gelu(
      mx::add(mx::conv1d(features,
                         checkpoint_.Weight("model.audio.conv1.weight"), 1, 1),
              checkpoint_.Weight("model.audio.conv1.bias")));
  hidden = qwen3_asr::Gelu(mx::add(
      mx::conv1d(hidden, checkpoint_.Weight("model.audio.conv2.weight"), 2, 1),
      checkpoint_.Weight("model.audio.conv2.bias")));
  const int length = hidden.shape(1);
  const int width = config_.audio_width;
  hidden = mx::astype(
      mx::add(hidden,
              mx::slice(checkpoint_.Weight("model.audio.positional_embedding"),
                        {0, 0}, {length, width})),
      hidden.dtype());
  for (int layer = 0; layer < config_.audio_layer_count; ++layer) {
    if (cancel.load()) {
      throw qwen3_asr::TranscriptionCancelled();
    } else {
    }
    const std::string prefix = "model.audio.blocks." + std::to_string(layer);
    const auto normalized =
        checkpoint_.LayerNorm(hidden, prefix + ".attn_ln", 1e-5f);
    std::vector<mx::array> projected;
    for (const std::string name : {"query", "key", "value"}) {
      projected.push_back(mx::transpose(
          mx::reshape(checkpoint_.Linear(normalized, prefix + ".attn." + name),
                      {1, length, config_.audio_head_count,
                       width / config_.audio_head_count}),
          {0, 2, 1, 3}));
    }
    auto attended = mx::fast::scaled_dot_product_attention(
        projected[0], projected[1], projected[2],
        1.0f / std::sqrt(static_cast<float>(width / config_.audio_head_count)));
    attended =
        mx::reshape(mx::transpose(attended, {0, 2, 1, 3}), {1, length, width});
    hidden =
        mx::add(hidden, checkpoint_.Linear(attended, prefix + ".attn.out"));
    const auto feed_forward = qwen3_asr::Gelu(checkpoint_.Linear(
        checkpoint_.LayerNorm(hidden, prefix + ".mlp_ln", 1e-5f),
        prefix + ".mlp.0"));
    hidden =
        mx::add(hidden, checkpoint_.Linear(feed_forward, prefix + ".mlp.2"));
    mx::eval(hidden);
  }
  hidden = checkpoint_.LayerNorm(hidden, "model.audio.ln_post", 1e-5f);
  hidden = qwen3_asr::Gelu(mx::add(
      mx::conv1d(hidden, checkpoint_.Weight("linear_proj_audio.0.weight"), 2,
                 1),
      checkpoint_.Weight("linear_proj_audio.0.bias")));
  hidden = checkpoint_.Linear(hidden, "linear_proj_audio.3");
  if (config_.normalize_audio) {
    const auto norm = mx::sqrt(mx::sum(
        mx::power(mx::astype(hidden, mx::float32), mx::array(2.0f)), -1, true));
    hidden =
        mx::astype(mx::divide(hidden, mx::maximum(norm, mx::array(1e-12f))),
                   features.dtype());
  } else {
  }
  mx::eval(hidden);
  return hidden;
}

mx::array MingModel::Encode(const std::vector<float> &samples,
                            const std::atomic<bool> &cancel) const {
  const auto dtype = checkpoint_.Weight("model.audio.conv1.weight").dtype();
  const auto features =
      mx::expand_dims(mx::astype(AudioFeatures(samples), dtype), 0);
  std::vector<mx::array> chunks;
  for (int start = 0; start < features.shape(1);
       start += config_.audio_chunk_frames) {
    chunks.push_back(EncodeChunk(
        mx::slice(
            features, {0, start, 0},
            {1, std::min(start + config_.audio_chunk_frames, features.shape(1)),
             128}),
        cancel));
  }
  auto encoded = mx::concatenate(chunks, 1);
  assert(encoded.shape(1) == config_.AudioTokenCount(features.shape(1)));
  return encoded;
}

} // namespace ming3_asr
