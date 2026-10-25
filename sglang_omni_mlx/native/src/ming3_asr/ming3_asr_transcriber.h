#pragma once

#include <atomic>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "common/tokenizer.h"
#include "ming3_asr/ming3_asr_model.h"
#include "qwen3_asr/qwen3_asr_transcriber.h"

namespace ming3_asr {

struct MingOptions {
  std::optional<std::string> language;
  int max_new_tokens = 256;
  int no_repeat_ngram_size = 10;
};

struct MingRequest {
  std::vector<float> samples;
  std::vector<int> suffix_ids;
  std::optional<std::string> language;
  int max_new_tokens;
  int no_repeat_ngram_size;
};

class MingTranscriber {
public:
  explicit MingTranscriber(const std::filesystem::path &model_directory);

  MingRequest Prepare(std::vector<float> samples,
                      const MingOptions &options) const;

  qwen3_asr::TranscriptionResult
  Transcribe(const MingRequest &request, const std::atomic<bool> &cancel) const;

private:
  MingModel model_;
  qwen3_asr::Tokenizer tokenizer_;
  std::vector<int> prefix_ids_;
  int audio_end_id_;
  int role_end_id_;
  int end_of_text_id_;
};

} // namespace ming3_asr
