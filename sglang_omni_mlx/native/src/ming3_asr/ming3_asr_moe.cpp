#include "ming3_asr/ming3_asr_model.h"

#include <algorithm>
#include <limits>

namespace ming3_asr {

namespace mx = mlx::core;

mx::array MingModel::DenseMlp(const mx::array &hidden,
                              const std::string &prefix) const {
  return checkpoint_.Linear(
      mx::multiply(
          qwen3_asr::Silu(checkpoint_.Linear(hidden, prefix + ".gate_proj")),
          checkpoint_.Linear(hidden, prefix + ".up_proj")),
      prefix + ".down_proj");
}

mx::array MingModel::FeedForward(const mx::array &hidden,
                                 const std::string &prefix, int layer) const {
  if (layer < config_.dense_layer_count) {
    return DenseMlp(hidden, prefix);
  } else {
  }
  const int length = hidden.shape(1);
  const int groups = config_.expert_group_count;
  const int per_group = config_.expert_count / groups;
  const int active = config_.active_expert_count;
  const auto scores = mx::sigmoid(mx::matmul(
      mx::astype(hidden, mx::float32),
      mx::transpose(mx::astype(checkpoint_.Weight(prefix + ".gate.weight"),
                               mx::float32))));
  auto adjusted =
      mx::add(scores, checkpoint_.Weight(prefix + ".gate.expert_bias"));
  if (groups > 1) {
    const auto grouped = mx::reshape(adjusted, {1, length, groups, per_group});
    const auto group_scores =
        mx::sum(mx::topk(grouped, std::min(2, per_group), -1), -1);
    const auto chosen = mx::slice(
        mx::argpartition(group_scores, -config_.active_expert_group_count, -1),
        {0, 0, groups - config_.active_expert_group_count},
        {1, length, groups});
    const auto allowed =
        mx::any(mx::equal(mx::arange(groups), mx::expand_dims(chosen, -1)), -2);
    adjusted = mx::reshape(
        mx::where(mx::expand_dims(allowed, -1), grouped,
                  mx::array(-std::numeric_limits<float>::infinity())),
        scores.shape());
  } else {
  }
  const auto selected = mx::slice(mx::argpartition(adjusted, -active, -1),
                                  {0, 0, config_.expert_count - active},
                                  {1, length, config_.expert_count});
  auto routing_weights = mx::take_along_axis(scores, selected, -1);
  if (active > 1) {
    routing_weights =
        mx::divide(routing_weights, mx::add(mx::sum(routing_weights, -1, true),
                                            mx::array(1e-20f)));
  } else {
  }
  routing_weights =
      mx::multiply(routing_weights, mx::array(config_.routed_scaling_factor));
  auto input = mx::reshape(hidden, {1, length, 1, 1, config_.hidden_size});
  auto indices = selected;
  std::optional<mx::array> inverse_order;
  const bool sorted = selected.size() >= 64;
  if (sorted) {
    const auto flattened = mx::reshape(selected, {-1});
    const auto order = mx::argsort(flattened);
    inverse_order = mx::argsort(order);
    indices = mx::take(flattened, order, 0);
    input =
        mx::take(mx::reshape(input, {length, 1, config_.hidden_size}),
                 mx::floor_divide(order, mx::array(active, order.dtype())), 0);
  } else {
  }
  const auto gate = checkpoint_.GatherLinear(
      input, prefix + ".switch_mlp.gate_proj", indices, sorted);
  const auto up = checkpoint_.GatherLinear(
      input, prefix + ".switch_mlp.up_proj", indices, sorted);
  const auto activated = mx::multiply(qwen3_asr::Silu(gate), up);
  auto routed = mx::squeeze(
      checkpoint_.GatherLinear(activated, prefix + ".switch_mlp.down_proj",
                               indices, sorted),
      -2);
  if (sorted) {
    routed = mx::reshape(mx::take(routed, *inverse_order, 0),
                         {1, length, active, config_.hidden_size});
  } else {
  }
  const auto output =
      mx::astype(mx::sum(mx::multiply(mx::astype(routed, mx::float32),
                                      mx::expand_dims(routing_weights, -1)),
                         -2),
                 hidden.dtype());
  return mx::add(output, DenseMlp(hidden, prefix + ".shared_experts"));
}

} // namespace ming3_asr
