#include "ming3_asr/ming3_asr_model.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "qwen3_asr/qwen3_asr_transcriber.h"

namespace ming3_asr {

namespace mx = mlx::core;

MingConfig::MingConfig(const nlohmann::json &config) {
  const auto &text = config.at("text_config");
  const auto &audio = config.at("audio_config");
  hidden_size = text.at("hidden_size");
  vocabulary_size = text.at("vocab_size");
  intermediate_size = text.at("intermediate_size");
  expert_intermediate_size = text.at("moe_intermediate_size");
  shared_intermediate_size = text.at("moe_shared_expert_intermediate_size");
  layer_count = text.at("num_hidden_layers");
  head_count = text.at("num_attention_heads");
  head_size = text.at("head_dim");
  layer_group_size = text.at("layer_group_size");
  kv_lora_rank = text.at("kv_lora_rank");
  query_lora_rank = text.at("q_lora_rank");
  rope_head_size = text.at("qk_rope_head_dim");
  query_head_size = text.at("qk_nope_head_dim");
  value_head_size = text.at("v_head_dim");
  convolution_kernel_size = text.at("short_conv_kernel_size");
  expert_count = text.at("num_experts");
  active_expert_count = text.at("num_experts_per_tok");
  expert_group_count = text.at("n_group");
  active_expert_group_count = text.at("topk_group");
  dense_layer_count = text.at("first_k_dense_replace");
  max_context_length = text.at("max_position_embeddings");
  eos_token_id = text.at("eos_token_id");
  audio_token_id = config.at("audio_token_id");
  audio_width = audio.at("n_audio_state");
  audio_head_count = audio.at("n_audio_head");
  audio_layer_count = audio.at("n_audio_layer");
  audio_context_length = audio.at("n_audio_ctx");
  audio_chunk_frames = audio.value("whisper_chunk_size", 3000);
  norm_epsilon = text.at("rms_norm_eps");
  rope_theta = text.at("rope_theta");
  routed_scaling_factor = text.at("routed_scaling_factor");
  kda_lower_bound = text.at("kda_lower_bound");
  normalize_audio = config.value("norm_query_embeds", true);
  const auto require = [](bool condition, const char *message) {
    if (!condition) {
      throw std::invalid_argument(message);
    } else {
    }
  };
  if (config.contains("quantization")) {
    const auto &quantized = config.at("quantization");
    quantization = {quantized.at("group_size"), quantized.at("bits"),
                    quantized.value("mode", std::string("affine"))};
    require(quantization.group_size == 64 && quantization.bits == 4 &&
                quantization.mode == "affine",
            "Ming ASR requires affine 4-bit weights with group_size=64");
  } else {
    throw std::invalid_argument("Ming ASR checkpoint is missing quantization");
  }
  require(config.at("model_type") == "ming_v3_asr",
          "Ming ASR model_type must be ming_v3_asr");
  require(text.at("no_kda_lora").get<bool>(),
          "Ming ASR requires no_kda_lora=true");
  require(text.at("kda_safe_gate").get<bool>(),
          "Ming ASR requires kda_safe_gate=true");
  require(text.at("rope_interleave").get<bool>(),
          "Ming ASR requires rope_interleave=true");
  require(text.at("hidden_act") == "silu",
          "Ming ASR requires hidden_act=silu");
  require(text.at("use_qkv_bias") == false,
          "Ming ASR requires use_qkv_bias=false");
  require(text.at("use_bias") == false, "Ming ASR requires use_bias=false");
  require(text.at("tie_word_embeddings") == false,
          "Ming ASR requires tie_word_embeddings=false");
  require(text.at("gated_attention_proj_granularity_type") == "head_wise",
          "Ming ASR requires head-wise attention projections");
  require(text.at("num_shared_experts") == 1,
          "Ming ASR requires one shared expert");
  require(!text.at("q_lora_rank").is_null(),
          "Ming ASR requires q_lora_rank");
  require(text.at("rope_scaling").is_null(),
          "Ming ASR requires rope_scaling=null");
  require(text.at("scoring_func") == "sigmoid",
          "Ming ASR requires scoring_func=sigmoid");
  require(!text.contains("expert_swiglu_limit_list"),
          "Ming ASR does not support expert_swiglu_limit_list");
  require(!text.contains("share_expert_swiglu_limit_list"),
          "Ming ASR does not support share_expert_swiglu_limit_list");
  require(audio.at("n_mels") == 128, "Ming ASR requires n_mels=128");
  require(audio.at("downsampler_kernel_size_list") ==
              nlohmann::json::array({3}),
          "Ming ASR requires one downsampler kernel of size 3");
  require(audio.at("downsampler_stride_list") == nlohmann::json::array({2}),
          "Ming ASR requires one downsampler stride of 2");
  require(audio.value("downsampler_padding_list", nlohmann::json::array({1})) ==
              nlohmann::json::array({1}),
          "Ming ASR requires one downsampler padding of 1");
  const std::vector<int> dimensions = {hidden_size,
                                       vocabulary_size,
                                       intermediate_size,
                                       expert_intermediate_size,
                                       shared_intermediate_size,
                                       layer_count,
                                       head_count,
                                       head_size,
                                       layer_group_size,
                                       kv_lora_rank,
                                       query_lora_rank,
                                       rope_head_size,
                                       query_head_size,
                                       value_head_size,
                                       convolution_kernel_size,
                                       expert_count,
                                       active_expert_count,
                                       expert_group_count,
                                       active_expert_group_count,
                                       max_context_length,
                                       audio_width,
                                       audio_head_count,
                                       audio_layer_count,
                                       audio_context_length,
                                       audio_chunk_frames};
  if (*std::min_element(dimensions.begin(), dimensions.end()) < 1 ||
      head_size % 32 != 0 || rope_head_size % 2 != 0 ||
      expert_count % expert_group_count != 0 ||
      active_expert_group_count > expert_group_count ||
      active_expert_count >
          active_expert_group_count * (expert_count / expert_group_count) ||
      audio_width % audio_head_count != 0 ||
      audio_chunk_frames > 2 * audio_context_length || dense_layer_count < 0 ||
      dense_layer_count > layer_count || eos_token_id < 0 ||
      eos_token_id >= vocabulary_size || audio_token_id < 0 ||
      audio_token_id >= vocabulary_size || !(norm_epsilon > 0) ||
      !std::isfinite(norm_epsilon) || !(rope_theta > 0) ||
      !std::isfinite(rope_theta) || !(kda_lower_bound < 0) ||
      !std::isfinite(kda_lower_bound) || !(routed_scaling_factor > 0) ||
      !std::isfinite(routed_scaling_factor)) {
    throw std::invalid_argument("invalid Ming ASR dimensions");
  } else {
  }
}

bool MingConfig::UsesMla(int layer) const {
  return (layer + 1) % layer_group_size == 0 ||
         layer >= layer_count / layer_group_size * layer_group_size;
}

int MingConfig::AudioTokenCount(int frames) const {
  int tokens = 0;
  for (int start = 0; start < frames; start += audio_chunk_frames) {
    const int chunk_frames = std::min(audio_chunk_frames, frames - start);
    tokens += ((chunk_frames + 1) / 2 + 1) / 2;
  }
  return tokens;
}

MingModel::MingModel(const std::filesystem::path &model_directory)
    : MingModel(qwen3_asr::ReadJson(model_directory / "config.json"),
                model_directory) {}

MingModel::MingModel(const nlohmann::json &configuration,
                     const std::filesystem::path &model_directory)
    : config_(configuration),
      checkpoint_(qwen3_asr::LoadSafetensors(model_directory), configuration) {
  ValidateWeights();
}

void MingModel::ValidateWeights() const {
  const auto expect = [&](const std::string &name, const mx::Shape &shape) {
    if (!checkpoint_.Has(name)) {
      throw std::invalid_argument("Ming checkpoint missing weight: " + name);
    } else if (checkpoint_.Weight(name).shape() != shape) {
      throw std::invalid_argument(
          "Ming checkpoint " + name + " shape: expected " +
          nlohmann::json(shape).dump() + ", got " +
          nlohmann::json(checkpoint_.Weight(name).shape()).dump());
    } else {
    }
  };
  const auto linear = [&](const std::string &prefix, int input, int output,
                          int experts = 0, bool bias = false) {
    mx::Shape shape = experts > 0 ? mx::Shape{experts, output, input}
                                  : mx::Shape{output, input};
    if (checkpoint_.Has(prefix + ".scales")) {
      if (config_.quantization.bits == 0 ||
          input % config_.quantization.group_size != 0 ||
          checkpoint_.Weight(prefix + ".weight").dtype() != mx::uint32) {
        throw std::invalid_argument("invalid Ming quantized projection: " +
                                    prefix);
      } else {
      }
      shape.back() = input / config_.quantization.group_size;
      expect(prefix + ".scales", shape);
      expect(prefix + ".biases", shape);
      shape.back() = input / (32 / config_.quantization.bits);
    } else {
    }
    expect(prefix + ".weight", shape);
    if (bias) {
      expect(prefix + ".bias", {output});
    } else {
    }
  };
  const int hidden = config_.hidden_size;
  const int audio = config_.audio_width;
  const int channels = config_.head_count * config_.head_size;
  linear("model.word_embeddings", hidden, config_.vocabulary_size);
  linear("lm_head", hidden, config_.vocabulary_size);
  expect("model.norm.weight", {hidden});
  expect("model.audio.conv1.weight", {audio, 3, 128});
  expect("model.audio.conv1.bias", {audio});
  expect("model.audio.conv2.weight", {audio, 3, audio});
  expect("model.audio.conv2.bias", {audio});
  expect("model.audio.positional_embedding",
         {config_.audio_context_length, audio});
  for (int layer = 0; layer < config_.audio_layer_count; ++layer) {
    const std::string prefix = "model.audio.blocks." + std::to_string(layer);
    for (const std::string norm : {"attn_ln", "mlp_ln"}) {
      expect(prefix + "." + norm + ".weight", {audio});
      expect(prefix + "." + norm + ".bias", {audio});
    }
    for (const std::string name : {"query", "key", "value", "out"}) {
      linear(prefix + ".attn." + name, audio, audio, 0, name != "key");
    }
    linear(prefix + ".mlp.0", audio, 4 * audio, 0, true);
    linear(prefix + ".mlp.2", 4 * audio, audio, 0, true);
  }
  expect("model.audio.ln_post.weight", {audio});
  expect("model.audio.ln_post.bias", {audio});
  expect("linear_proj_audio.0.weight", {hidden, 3, audio});
  expect("linear_proj_audio.0.bias", {hidden});
  linear("linear_proj_audio.3", hidden, hidden, 0, true);
  for (int layer = 0; layer < config_.layer_count; ++layer) {
    const std::string prefix = "model.layers." + std::to_string(layer);
    expect(prefix + ".input_layernorm.weight", {hidden});
    expect(prefix + ".post_attention_layernorm.weight", {hidden});
    const std::string attention = prefix + ".attention";
    if (config_.UsesMla(layer)) {
      linear(attention + ".q_a_proj", hidden, config_.query_lora_rank);
      expect(attention + ".q_a_layernorm.weight", {config_.query_lora_rank});
      linear(attention + ".q_b_proj", config_.query_lora_rank,
             config_.head_count *
                 (config_.query_head_size + config_.rope_head_size));
      linear(attention + ".kv_a_proj_with_mqa", hidden,
             config_.kv_lora_rank + config_.rope_head_size);
      expect(attention + ".kv_a_layernorm.weight", {config_.kv_lora_rank});
      linear(attention + ".kv_b_proj", config_.kv_lora_rank,
             config_.head_count *
                 (config_.query_head_size + config_.value_head_size));
      linear(attention + ".g_proj", hidden, config_.head_count);
      linear(attention + ".dense", config_.head_count * config_.value_head_size,
             hidden);
    } else {
      for (const std::string name : {"q", "k", "v"}) {
        linear(attention + "." + name + "_proj", hidden, channels);
        expect(attention + "." + name + "_conv1d.weight",
               {channels, config_.convolution_kernel_size, 1});
      }
      linear(attention + ".b_proj", hidden, config_.head_count);
      linear(attention + ".f_proj", hidden, channels);
      linear(attention + ".g_proj", hidden, channels);
      linear(attention + ".o_proj", channels, hidden);
      expect(attention + ".A_log", {config_.head_count});
      expect(attention + ".dt_bias", {channels});
      expect(attention + ".o_norm.weight", {config_.head_size});
    }
    const std::string feed_forward = prefix + ".mlp";
    if (layer < config_.dense_layer_count) {
      linear(feed_forward + ".gate_proj", hidden, config_.intermediate_size);
      linear(feed_forward + ".up_proj", hidden, config_.intermediate_size);
      linear(feed_forward + ".down_proj", config_.intermediate_size, hidden);
    } else {
      expect(feed_forward + ".gate.weight", {config_.expert_count, hidden});
      expect(feed_forward + ".gate.expert_bias", {config_.expert_count});
      for (const std::string name : {"gate_proj", "up_proj"}) {
        linear(feed_forward + ".switch_mlp." + name, hidden,
               config_.expert_intermediate_size, config_.expert_count);
        linear(feed_forward + ".shared_experts." + name, hidden,
               config_.shared_intermediate_size);
      }
      linear(feed_forward + ".switch_mlp.down_proj",
             config_.expert_intermediate_size, hidden, config_.expert_count);
      linear(feed_forward + ".shared_experts.down_proj",
             config_.shared_intermediate_size, hidden);
    }
  }
}

mx::array MingModel::Embed(const std::vector<int> &ids) const {
  return checkpoint_.Embed(
      mx::array(ids.data(), {1, static_cast<int>(ids.size())}),
      "model.word_embeddings");
}

MingCache::MingCache(const MingModel *owner) : owner_(owner) {
  layers_.reserve(owner->config().layer_count);
  for (int layer = 0; layer < owner->config().layer_count; ++layer) {
    if (owner->config().UsesMla(layer)) {
      layers_.emplace_back(LatentAttentionCache{});
    } else {
      layers_.emplace_back(DeltaCache{});
    }
  }
}

MingCache MingModel::NewCache() const { return MingCache(this); }

mx::array MingModel::Forward(const mx::array &embeddings,
                             MingCache &cache,
                             const std::atomic<bool> &cancel) const {
  if (cache.owner_ != this || cache.layers_.size() !=
                                  static_cast<size_t>(config_.layer_count)) {
    throw std::invalid_argument("Ming cache does not belong to this model");
  } else if (!cache.is_valid_) {
    throw std::invalid_argument(
        "Ming cache is invalid after a failed or cancelled Forward; create a "
        "new cache");
  } else if (embeddings.ndim() != 3 || embeddings.shape(0) != 1 ||
             embeddings.shape(1) < 1 ||
             embeddings.shape(2) != config_.hidden_size) {
    throw std::invalid_argument(
        "Ming Forward embeddings must have shape [1, positive token_count, " +
        std::to_string(config_.hidden_size) + "]");
  } else if (embeddings.shape(1) >
             config_.max_context_length - cache.token_count_) {
    throw std::invalid_argument("Ming Forward exceeds max_position_embeddings");
  } else {
  }
  cache.is_valid_ = false;
  mx::array hidden = embeddings;
  std::vector<mx::array> evaluated;
  for (int layer = 0; layer < config_.layer_count; ++layer) {
    if (cancel.load()) {
      throw qwen3_asr::TranscriptionCancelled();
    } else {
    }
    const std::string prefix = "model.layers." + std::to_string(layer);
    auto normalized = checkpoint_.RmsNorm(hidden, prefix + ".input_layernorm",
                                          config_.norm_epsilon);
    hidden =
        mx::add(hidden, config_.UsesMla(layer)
                            ? LatentAttention(
                                  normalized, prefix + ".attention",
                                  std::get<LatentAttentionCache>(
                                      cache.layers_.at(layer)))
                            : DeltaAttention(
                                  normalized, prefix + ".attention",
                                  std::get<DeltaCache>(cache.layers_.at(layer))));
    normalized = checkpoint_.RmsNorm(
        hidden, prefix + ".post_attention_layernorm", config_.norm_epsilon);
    hidden = mx::add(hidden, FeedForward(normalized, prefix + ".mlp", layer));
    const auto &layer_cache = cache.layers_.at(layer);
    if (const auto *delta = std::get_if<DeltaCache>(&layer_cache)) {
      const auto &state = delta->state.value();
      evaluated.push_back(state.recurrent);
      for (const auto &convolution : state.convolution) {
        evaluated.push_back(convolution);
      }
    } else {
      const auto &state =
          std::get<LatentAttentionCache>(layer_cache).state.value();
      evaluated.push_back(state.keys);
      evaluated.push_back(state.values);
    }
  }
  hidden = checkpoint_.RmsNorm(hidden, "model.norm", config_.norm_epsilon);
  evaluated.push_back(hidden);
  mx::eval(evaluated);
  cache.token_count_ += embeddings.shape(1);
  cache.is_valid_ = true;
  return hidden;
}

mx::array MingModel::Logits(const mx::array &hidden) const {
  return mx::astype(
      checkpoint_.Linear(
          mx::slice(hidden, {0, hidden.shape(1) - 1, 0}, hidden.shape()),
          "lm_head"),
      mx::float32);
}

} // namespace ming3_asr
