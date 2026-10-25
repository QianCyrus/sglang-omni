#include "ming3_asr/ming3_asr_transcriber.h"

#include <algorithm>
#include <cctype>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>

#include "common/audio.h"

namespace ming3_asr {

namespace mx = mlx::core;

namespace {

constexpr int kPrefillChunkSize = 256;
constexpr const char *kInstruction =
    "Please recognize the language of this speech and transcribe it, "
    "placing a `\t` between the identified language and the transcribe.";

std::optional<std::string>
LanguageCode(const std::optional<std::string> &language) {
  if (!language.has_value() || language->empty()) {
    return std::nullopt;
  } else {
    std::string hint = qwen3_asr::StripUnicodeWhitespace(*language);
    std::transform(hint.begin(), hint.end(), hint.begin(),
                   [](unsigned char character) {
                     return static_cast<char>(std::tolower(character));
                   });
    if (hint == "en" || hint == "english") {
      return "en";
    } else if (hint == "zh" || hint == "chinese") {
      return "zh";
    } else {
      throw std::invalid_argument("Ming ASR language must be en or zh");
    }
  }
}

} // namespace

MingTranscriber::MingTranscriber(const std::filesystem::path &model_directory)
    : model_(model_directory), tokenizer_(model_directory),
      prefix_ids_(tokenizer_.Encode(std::string(kInstruction) + "<audio>")),
      audio_end_id_(tokenizer_.AddedTokenId("</audio>")),
      role_end_id_(tokenizer_.AddedTokenId("<|role_end|>")),
      end_of_text_id_(tokenizer_.AddedTokenId("<|endoftext|>")) {
  if (tokenizer_.AddedTokenId("<audioPatch>") !=
          model_.config().audio_token_id ||
      tokenizer_.AddedTokenId("<audio>") < 0 || audio_end_id_ < 0 ||
      role_end_id_ < 0 || end_of_text_id_ < 0) {
    throw std::invalid_argument(
        "Ming checkpoint is missing required audio or stop tokens");
  } else {
  }
}

MingRequest MingTranscriber::Prepare(std::vector<float> samples,
                                     const MingOptions &options) const {
  if (options.max_new_tokens < 1 || options.no_repeat_ngram_size < 0) {
    throw std::invalid_argument(
        "max_new_tokens must be positive and no_repeat_ngram_size nonnegative");
  } else {
  }
  ValidateAudio(samples);
  const auto language = LanguageCode(options.language);
  std::vector<int> suffix_ids{audio_end_id_};
  if (language.has_value()) {
    const auto hint_ids =
        tokenizer_.Encode(*language == "en" ? "English\t" : "Chinese\t");
    suffix_ids.insert(suffix_ids.end(), hint_ids.begin(), hint_ids.end());
  } else {
  }
  const auto &config = model_.config();
  const size_t frame_count = samples.size() / qwen3_asr::kHopLength;
  if (frame_count > static_cast<size_t>(config.max_context_length) * 4) {
    throw std::invalid_argument("audio exceeds the Ming ASR context budget");
  } else {
  }
  const int audio_tokens =
      config.AudioTokenCount(static_cast<int>(frame_count));
  const int prompt_tokens =
      static_cast<int>(prefix_ids_.size() + suffix_ids.size()) + audio_tokens;
  const int available = config.max_context_length - prompt_tokens;
  if (available < 1) {
    throw std::invalid_argument(
        "audio prompt exceeds the Ming ASR context budget");
  } else {
  }
  return {std::move(samples), std::move(suffix_ids), language,
          std::min(options.max_new_tokens, available),
          options.no_repeat_ngram_size};
}

qwen3_asr::TranscriptionResult
MingTranscriber::Transcribe(const MingRequest &request,
                            const std::atomic<bool> &cancel) const {
  if (cancel.load()) {
    throw qwen3_asr::TranscriptionCancelled();
  } else {
  }
  const auto &config = model_.config();
  const auto prefix = model_.Embed(prefix_ids_);
  const auto audio =
      mx::astype(model_.Encode(request.samples, cancel), prefix.dtype());
  const auto embeddings =
      mx::concatenate({prefix, audio, model_.Embed(request.suffix_ids)}, 1);
  auto cache = model_.NewCache();
  std::optional<mx::array> hidden;
  for (int start = 0; start < embeddings.shape(1); start += kPrefillChunkSize) {
    hidden = model_.Forward(
        mx::slice(embeddings, {0, start, 0},
                  {1, std::min(start + kPrefillChunkSize, embeddings.shape(1)),
                   config.hidden_size}),
        cache, cancel);
  }
  std::vector<int> output_ids;
  qwen3_asr::TranscriptionResult result;
  for (int step = 0; step < request.max_new_tokens; ++step) {
    if (cancel.load()) {
      throw qwen3_asr::TranscriptionCancelled();
    } else {
    }
    auto logits = mx::reshape(model_.Logits(hidden.value()), {-1});
    const size_t ngram = static_cast<size_t>(request.no_repeat_ngram_size);
    if (ngram > 0 && output_ids.size() >= ngram) {
      std::vector<int> banned;
      for (size_t start = 0; start + ngram <= output_ids.size(); ++start) {
        if (std::equal(output_ids.begin() + start,
                       output_ids.begin() + start + ngram - 1,
                       output_ids.end() - (ngram - 1))) {
          banned.push_back(output_ids[start + ngram - 1]);
        } else {
        }
      }
      if (!banned.empty()) {
        logits = mx::put_along_axis(
            logits, mx::array(banned.data(), {static_cast<int>(banned.size())}),
            mx::array(-std::numeric_limits<float>::infinity()), 0);
      } else {
      }
    } else {
    }
    const int token = mx::argmax(logits).item<int>();
    if (token == config.eos_token_id || token == role_end_id_ ||
        token == end_of_text_id_) {
      result.finish_reason = qwen3_asr::FinishReason::kStop;
      break;
    } else {
      output_ids.push_back(token);
    }
    if (step + 1 < request.max_new_tokens) {
      hidden = model_.Forward(model_.Embed({token}), cache, cancel);
    } else {
    }
  }
  result.text =
      qwen3_asr::StripUnicodeWhitespace(tokenizer_.Decode(output_ids, true));
  result.language = request.language;
  static const std::map<std::string, std::string> language_codes = {
      {"English", "en"},
      {"Chinese", "zh"},
      {"Canton", "yue"},
      {"Japanese", "ja"},
      {"Korean", "ko"}};
  const size_t separator = result.text.find('\t');
  if (separator != std::string::npos) {
    const std::string label =
        qwen3_asr::StripUnicodeWhitespace(result.text.substr(0, separator));
    const auto found = language_codes.find(label);
    result.language = found == language_codes.end() ? label : found->second;
    result.text =
        qwen3_asr::StripUnicodeWhitespace(result.text.substr(separator + 1));
  } else if (!request.language.has_value()) {
    for (const auto &[label, code] : language_codes) {
      if (result.text.starts_with(label)) {
        result.language = code;
        result.text =
            qwen3_asr::StripUnicodeWhitespace(result.text.substr(label.size()));
        break;
      } else {
      }
    }
  } else {
  }
  result.generated_token_count = static_cast<int>(output_ids.size());
  return result;
}

} // namespace ming3_asr
