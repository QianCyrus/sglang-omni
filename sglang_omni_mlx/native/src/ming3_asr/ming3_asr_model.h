#pragma once

#include <array>
#include <atomic>
#include <filesystem>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "common/layers.h"

namespace ming3_asr {

void ValidateAudio(const std::vector<float> &samples);
mlx::core::array AudioFeatures(const std::vector<float> &samples);

struct MingConfig {
  explicit MingConfig(const nlohmann::json &config);

  int hidden_size;
  int vocabulary_size;
  int intermediate_size;
  int expert_intermediate_size;
  int shared_intermediate_size;
  int layer_count;
  int head_count;
  int head_size;
  int layer_group_size;
  int kv_lora_rank;
  int query_lora_rank;
  int rope_head_size;
  int query_head_size;
  int value_head_size;
  int convolution_kernel_size;
  int expert_count;
  int active_expert_count;
  int expert_group_count;
  int active_expert_group_count;
  int dense_layer_count;
  int max_context_length;
  int eos_token_id;
  int audio_token_id;
  int audio_width;
  int audio_head_count;
  int audio_layer_count;
  int audio_context_length;
  int audio_chunk_frames;
  float norm_epsilon;
  float rope_theta;
  float routed_scaling_factor;
  float kda_lower_bound;
  bool normalize_audio;
  qwen3_asr::QuantizationConfig quantization;

  bool UsesMla(int layer) const;
  int AudioTokenCount(int frames) const;
};

struct DeltaState {
  std::array<mlx::core::array, 3> convolution;
  mlx::core::array recurrent;
};

struct LatentAttentionState {
  mlx::core::array keys;
  mlx::core::array values;
};

struct DeltaCache {
  std::optional<DeltaState> state;
};

struct LatentAttentionCache {
  std::optional<LatentAttentionState> state;
};

using LayerCache = std::variant<DeltaCache, LatentAttentionCache>;

class MingModel;

class MingCache {
public:
  const std::vector<LayerCache> &layers() const { return layers_; }

private:
  friend class MingModel;
  explicit MingCache(const MingModel *owner);

  const MingModel *owner_;
  std::vector<LayerCache> layers_;
  int token_count_ = 0;
  bool is_valid_ = true;
};

struct GatedDeltaResult {
  mlx::core::array output;
  mlx::core::array next_state;
};

GatedDeltaResult GatedDelta(
    const mlx::core::array &queries, const mlx::core::array &keys,
    const mlx::core::array &values, const mlx::core::array &decay,
    const mlx::core::array &beta, const mlx::core::array &state);

class MingModel {
public:
  explicit MingModel(const std::filesystem::path &model_directory);

  const MingConfig &config() const { return config_; }
  MingCache NewCache() const;
  mlx::core::array Embed(const std::vector<int> &ids) const;
  mlx::core::array Encode(const std::vector<float> &samples,
                          const std::atomic<bool> &cancel) const;
  mlx::core::array Forward(const mlx::core::array &embeddings,
                           MingCache &cache,
                           const std::atomic<bool> &cancel) const;
  mlx::core::array Logits(const mlx::core::array &hidden) const;

private:
  MingModel(const nlohmann::json &configuration,
            const std::filesystem::path &model_directory);
  void ValidateWeights() const;
  mlx::core::array EncodeChunk(const mlx::core::array &features,
                               const std::atomic<bool> &cancel) const;
  mlx::core::array DeltaAttention(const mlx::core::array &hidden,
                                  const std::string &prefix,
                                  DeltaCache &cache) const;
  mlx::core::array LatentAttention(const mlx::core::array &hidden,
                                   const std::string &prefix,
                                   LatentAttentionCache &cache) const;
  mlx::core::array FeedForward(const mlx::core::array &hidden,
                               const std::string &prefix, int layer) const;
  mlx::core::array DenseMlp(const mlx::core::array &hidden,
                            const std::string &prefix) const;

  MingConfig config_;
  qwen3_asr::Checkpoint checkpoint_;
};

} // namespace ming3_asr
