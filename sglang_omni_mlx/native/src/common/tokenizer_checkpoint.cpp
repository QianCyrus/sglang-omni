#include "common/tokenizer.h"

#include <algorithm>
#include <fstream>
#include <stdexcept>

#include "common/layers.h"

namespace qwen3_asr {

std::string
Tokenizer::LoadVocabulary(const std::filesystem::path &model_directory) {
  constexpr const char *kQwen2SplitPattern =
      R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N})"
      R"(| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
  nlohmann::json vocabulary;
  std::string split_pattern = kQwen2SplitPattern;
  if (std::filesystem::exists(model_directory / "vocab.json")) {
    vocabulary = ReadJson(model_directory / "vocab.json");
    std::ifstream merges(model_directory / "merges.txt");
    if (!merges) {
      throw std::runtime_error("cannot read " +
                               (model_directory / "merges.txt").string());
    } else {
    }
    std::string line;
    int rank = 0;
    while (std::getline(merges, line)) {
      if (line.rfind("#version", 0) == 0 || line.empty()) {
        continue;
      } else {
        merge_ranks_.emplace(line, rank++);
      }
    }
  } else {
    const auto tokenizer = ReadJson(model_directory / "tokenizer.json");
    const auto &model = tokenizer.at("model");
    const auto &pretokenizers =
        tokenizer.at("pre_tokenizer").at("pretokenizers");
    if (model.at("type") != "BPE" || !model.at("dropout").is_null() ||
        !model.at("unk_token").is_null() ||
        !model.at("continuing_subword_prefix").is_null() ||
        !model.at("end_of_word_suffix").is_null() ||
        model.at("byte_fallback") != false ||
        model.at("ignore_merges") != false ||
        tokenizer.at("normalizer").at("type") != "NFC" ||
        tokenizer.at("pre_tokenizer").at("type") != "Sequence" ||
        pretokenizers.size() != 2 ||
        pretokenizers.at(0).at("type") != "Split" ||
        pretokenizers.at(0).at("behavior") != "Isolated" ||
        pretokenizers.at(0).at("invert") != false ||
        pretokenizers.at(1).at("type") != "ByteLevel" ||
        pretokenizers.at(1).at("add_prefix_space") != false ||
        pretokenizers.at(1).at("use_regex") != false ||
        tokenizer.at("decoder").at("type") != "ByteLevel") {
      throw std::invalid_argument(
          "unsupported tokenizer.json byte-level BPE configuration");
    } else {
      vocabulary = model.at("vocab");
      split_pattern =
          pretokenizers.at(0).at("pattern").at("Regex").get<std::string>();
      int rank = 0;
      for (const auto &merge : model.at("merges")) {
        const std::string pair = merge.is_string()
                                     ? merge.get<std::string>()
                                     : merge.at(0).get<std::string>() + " " +
                                           merge.at(1).get<std::string>();
        merge_ranks_.emplace(pair, rank++);
      }
    }
  }
  for (const auto &[token, id] : vocabulary.items()) {
    vocabulary_.emplace(token, id.get<int>());
  }
  const auto config = ReadJson(model_directory / "tokenizer_config.json");
  if (config.contains("added_tokens_decoder")) {
    for (const auto &[id_text, token] :
         config.at("added_tokens_decoder").items()) {
      const int id = std::stoi(id_text);
      added_token_index_by_id_[id] = added_tokens_.size();
      added_tokens_.push_back({token.at("content").get<std::string>(), id,
                               token.at("special").get<bool>()});
    }
  } else {
    const auto tokenizer = ReadJson(model_directory / "tokenizer.json");
    for (const auto &token : tokenizer.at("added_tokens")) {
      const int id = token.at("id").get<int>();
      added_token_index_by_id_[id] = added_tokens_.size();
      added_tokens_.push_back({token.at("content").get<std::string>(), id,
                               token.at("special").get<bool>()});
    }
  }
  size_t largest_id = 0;
  for (const auto &[token, id] : vocabulary_) {
    if (id < 0) {
      throw std::invalid_argument("negative vocabulary token id");
    } else {
      largest_id = std::max(largest_id, static_cast<size_t>(id));
    }
  }
  for (const auto &added : added_tokens_) {
    if (added.id < 0 || added.content.empty()) {
      throw std::invalid_argument("invalid added token");
    } else {
      largest_id = std::max(largest_id, static_cast<size_t>(added.id));
    }
  }
  tokens_by_id_.resize(largest_id + 1);
  for (const auto &[token, id] : vocabulary_) {
    tokens_by_id_[id] = token;
  }
  return split_pattern;
}

} // namespace qwen3_asr
