#include "ming3_asr/ming3_asr_model.h"

#include <cmath>
#include <utility>

namespace ming3_asr {

namespace mx = mlx::core;

mx::array MingModel::DeltaAttention(const mx::array &hidden,
                                    const std::string &prefix,
                                    DeltaCache &cache) const {
  const int length = hidden.shape(1);
  const int heads = config_.head_count;
  const int width = config_.head_size;
  const int channels = heads * width;
  const int keep = config_.convolution_kernel_size - 1;
  std::array<mx::array, 3> convolution_history =
      cache.state.has_value()
          ? cache.state->convolution
          : std::array{mx::zeros({1, keep, channels}, hidden.dtype()),
                       mx::zeros({1, keep, channels}, hidden.dtype()),
                       mx::zeros({1, keep, channels}, hidden.dtype())};
  const auto project = [&](const std::string &name, size_t index) {
    auto value = checkpoint_.Linear(hidden, prefix + "." + name + "_proj");
    value = mx::concatenate({convolution_history[index], value}, 1);
    convolution_history[index] =
        mx::slice(value, {0, length, 0}, value.shape());
    value = qwen3_asr::Silu(mx::conv1d(
        value, checkpoint_.Weight(prefix + "." + name + "_conv1d.weight"), 1, 0,
        1, channels));
    return mx::reshape(value, {1, length, heads, width});
  };
  const std::array projected{project("q", 0), project("k", 1),
                             project("v", 2)};
  auto queries = mx::astype(projected[0], mx::float32);
  auto keys = mx::astype(projected[1], mx::float32);
  queries = mx::multiply(
      queries, mx::rsqrt(mx::add(mx::sum(mx::square(queries), -1, true),
                                 mx::array(1e-6f))));
  queries = mx::multiply(
      queries, mx::array(1.0f / std::sqrt(static_cast<float>(width))));
  keys =
      mx::multiply(keys, mx::rsqrt(mx::add(mx::sum(mx::square(keys), -1, true),
                                           mx::array(1e-6f))));
  const auto gate = mx::reshape(
      mx::add(mx::astype(checkpoint_.Linear(hidden, prefix + ".f_proj"),
                         mx::float32),
              checkpoint_.Weight(prefix + ".dt_bias")),
      {1, length, heads, width});
  const auto rate = mx::reshape(
      mx::exp(mx::astype(checkpoint_.Weight(prefix + ".A_log"), mx::float32)),
      {heads, 1});
  const auto decay =
      mx::exp(mx::multiply(mx::array(config_.kda_lower_bound),
                           mx::sigmoid(mx::multiply(rate, gate))));
  const auto beta = mx::sigmoid(
      mx::astype(checkpoint_.Linear(hidden, prefix + ".b_proj"), mx::float32));
  const auto state = cache.state.has_value()
                         ? cache.state->recurrent
                         : mx::zeros({1, heads, width, width}, mx::float32);
  const auto recurrent =
      GatedDelta(queries, keys, projected[2], decay, beta, state);
  cache.state = DeltaState{std::move(convolution_history), recurrent.next_state};
  auto output =
      mx::astype(mx::astype(recurrent.output, projected[2].dtype()),
                 mx::float32);
  output = mx::fast::rms_norm(
      output,
      mx::astype(checkpoint_.Weight(prefix + ".o_norm.weight"), mx::float32),
      config_.norm_epsilon);
  const auto output_gate =
      mx::astype(mx::reshape(checkpoint_.Linear(hidden, prefix + ".g_proj"),
                             output.shape()),
                 mx::float32);
  output =
      mx::reshape(mx::astype(mx::multiply(output, mx::sigmoid(output_gate)),
                             hidden.dtype()),
                  {1, length, channels});
  return checkpoint_.Linear(output, prefix + ".o_proj");
}

mx::array MingModel::LatentAttention(const mx::array &hidden,
                                     const std::string &prefix,
                                     LatentAttentionCache &cache) const {
  const int length = hidden.shape(1);
  const int heads = config_.head_count;
  const int offset = cache.state.has_value() ? cache.state->keys.shape(2) : 0;
  auto queries = checkpoint_.Linear(
      checkpoint_.RmsNorm(checkpoint_.Linear(hidden, prefix + ".q_a_proj"),
                          prefix + ".q_a_layernorm", config_.norm_epsilon),
      prefix + ".q_b_proj");
  queries =
      mx::transpose(mx::reshape(queries, {1, length, heads, -1}), {0, 2, 1, 3});
  const auto query_parts =
      mx::split(queries, mx::Shape{config_.query_head_size}, -1);
  const auto latent_parts =
      mx::split(checkpoint_.Linear(hidden, prefix + ".kv_a_proj_with_mqa"),
                mx::Shape{config_.kv_lora_rank}, -1);
  auto latent = checkpoint_.Linear(
      checkpoint_.RmsNorm(latent_parts[0], prefix + ".kv_a_layernorm",
                          config_.norm_epsilon),
      prefix + ".kv_b_proj");
  latent =
      mx::transpose(mx::reshape(latent, {1, length, heads, -1}), {0, 2, 1, 3});
  const auto key_value =
      mx::split(latent, mx::Shape{config_.query_head_size}, -1);
  const auto rotated_queries =
      mx::fast::rope(query_parts[1], config_.rope_head_size, true,
                     config_.rope_theta, 1.0f, offset);
  const auto rotated_keys = mx::fast::rope(mx::expand_dims(latent_parts[1], 1),
                                           config_.rope_head_size, true,
                                           config_.rope_theta, 1.0f, offset);
  queries = mx::concatenate({query_parts[0], rotated_queries}, -1);
  auto keys = mx::concatenate(
      {key_value[0], mx::broadcast_to(rotated_keys, rotated_queries.shape())},
      -1);
  auto values = key_value[1];
  if (cache.state.has_value()) {
    keys = mx::concatenate({cache.state->keys, keys}, 2);
    values = mx::concatenate({cache.state->values, values}, 2);
  } else {
  }
  cache.state = LatentAttentionState{keys, values};
  auto output = mx::fast::scaled_dot_product_attention(
      queries, keys, values,
      1.0f / std::sqrt(static_cast<float>(config_.query_head_size +
                                          config_.rope_head_size)),
      length > 1 ? "causal" : "");
  output = mx::transpose(output, {0, 2, 1, 3});
  const auto gate = mx::reshape(
      mx::astype(
          mx::sigmoid(mx::astype(checkpoint_.Linear(hidden, prefix + ".g_proj"),
                                 mx::float32)),
          hidden.dtype()),
      {1, length, heads, 1});
  return checkpoint_.Linear(
      mx::reshape(mx::multiply(output, gate),
                  {1, length, heads * config_.value_head_size}),
      prefix + ".dense");
}

} // namespace ming3_asr
