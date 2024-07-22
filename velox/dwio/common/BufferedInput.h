/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "velox/common/memory/AllocationPool.h"
#include "velox/dwio/common/SeekableInputStream.h"
#include "velox/dwio/common/StreamIdentifier.h"

// Use WS VRead API to load
DECLARE_bool(wsVRLoad);

namespace facebook::velox::dwio::common {

/// Velox 的读 API, dwio 读的 Base, 是一个味道非常奇怪的接口, 负责对同一个
/// "文件" ( 或者至少是 split 吧, 一个 split 最多一个 file?) 的 io.
///
/// 外部的请求按照 `Region` 来切分, Region 带上了对应的 Label.
class BufferedInput {
 public:
  // 默认合并 1.25MB 的距离
  constexpr static uint64_t kMaxMergeDistance = 1024 * 1024 * 1.25;

  BufferedInput(
      std::shared_ptr<ReadFile> readFile,
      memory::MemoryPool& pool,
      const MetricsLogPtr& metricsLog = MetricsLog::voidLog(),
      IoStatistics* stats = nullptr,
      uint64_t maxMergeDistance = kMaxMergeDistance,
      std::optional<bool> wsVRLoad = std::nullopt)
      : BufferedInput(
            std::make_shared<ReadFileInputStream>(
                std::move(readFile),
                metricsLog,
                stats),
            pool,
            maxMergeDistance,
            wsVRLoad) {}

  BufferedInput(
      std::shared_ptr<ReadFileInputStream> input,
      memory::MemoryPool& pool,
      uint64_t maxMergeDistance = kMaxMergeDistance,
      std::optional<bool> wsVRLoad = std::nullopt)
      : input_{std::move(input)},
        pool_{&pool},
        maxMergeDistance_{maxMergeDistance},
        wsVRLoad_{wsVRLoad},
        allocPool_{std::make_unique<memory::AllocationPool>(&pool)} {}

  BufferedInput(BufferedInput&&) = default;
  virtual ~BufferedInput() = default;

  BufferedInput(const BufferedInput&) = delete;
  BufferedInput& operator=(const BufferedInput&) = delete;
  BufferedInput& operator=(BufferedInput&&) = delete;

  virtual const std::string& getName() const {
    return input_->getName();
  }

  /// 这里面它拆分成了一个两阶段的 API, 先去 enqueue, 然后对 enqueue 的对象再去 Load.

  /// The previous API was taking a vector of regions. Now we allow callers to
  /// enqueue region any time/place and we do final load into buffer in 2 steps
  /// (enqueue....load). 'si' allows tracking which streams actually get read.
  /// This may control read-ahead and caching for BufferedInput implementations
  /// supporting these.
  virtual std::unique_ptr<SeekableInputStream> enqueue(
      velox::common::Region region,
      const StreamIdentifier* sid = nullptr);

  /// Returns true if load synchronously.
  ///
  /// https://github.com/facebookincubator/velox/commit/ca5e409aad91462b0f4280b5ee24358deb9f97a1
  /// 这个感觉就是额外开一个 Load 吧..
  virtual bool supportSyncLoad() const {
    return true;
  }

  /// load all regions to be read in an optimized way (IO efficiency)
  ///
  /// 内部去尝试全部 Load
  virtual void load(const LogType);

  /// check 一段 Range 是否被 buffered
  virtual bool isBuffered(uint64_t offset, uint64_t length) const {
    return !!readBuffer(offset, length);
  }

  virtual std::unique_ptr<SeekableInputStream>
  read(uint64_t offset, uint64_t length, LogType logType) const {
    std::unique_ptr<SeekableInputStream> ret = readBuffer(offset, length);
    if (ret != nullptr) {
      return ret;
    }
    VLOG(1) << "Unplanned read. Offset: " << offset << ", Length: " << length;
    // We cannot do enqueue/load here because load() clears previously
    // loaded data. TODO: figure out how we can use the data cache for
    // this access.
    return std::make_unique<SeekableFileInputStream>(
        input_, offset, length, *pool_, logType, input_->getNaturalReadSize());
  }

  // True if there is free memory for prefetching the stripe. This is
  // called to check if a stripe that is not next for read should be
  // prefetched. 'numPages' is added to the already enqueued pages, so
  // that this can be called also before enqueueing regions.
  virtual bool shouldPreload(int32_t /*numPages*/ = 0) {
    return false;
  }

  // True if caller should enqueue and load regions for stripe
  // metadata after reading a file footer. The stripe footers are
  // likely to be hit and should be read ahead of demand if
  // BufferedInput has background load.
  //
  // 在 IO 介质上判断这里的内容需要 Prefetch, 看实现上 Cached 就会需要
  // buffer. 否则如果是 Direct 就不去 Buffer.
  virtual bool shouldPrefetchStripes() const {
    return false;
  }

  virtual void setNumStripes(int32_t /*numStripes*/) {}

  // Create a new (clean) instance of BufferedInput sharing the same
  // underlying file and memory pool.  The enqueued regions are NOT copied.
  virtual std::unique_ptr<BufferedInput> clone() const {
    return std::make_unique<BufferedInput>(
        input_, *pool_, maxMergeDistance_, wsVRLoad_);
  }

  std::unique_ptr<SeekableInputStream> loadCompleteFile() {
    auto stream = enqueue({0, input_->getLength()});
    load(dwio::common::LogType::FILE);
    return stream;
  }

  const std::shared_ptr<ReadFile>& getReadFile() const {
    return input_->getReadFile();
  }

  // Internal API, do not use outside Velox.
  //
  // 因为这个接口是包了一层 ReadFile 的, 这里应该是类似直接拿到 ReadFile 或者 read
  // file 的 Stream 吧, 和上面那个 sb 接口 getReadFile 差不多.
  const std::shared_ptr<ReadFileInputStream>& getInputStream() const {
    return input_;
  }

  virtual folly::Executor* executor() const {
    return nullptr;
  }

  // 一个比较 sb 的接口, 感觉不是一个精确的估计, 这里实现成了所有 region
  // length 的和.
  // 操你妈的, 下面 Cache 和 Direct 都没实现这个接口, 我操了.
  virtual uint64_t nextFetchSize() const;

 protected:
  const std::shared_ptr<ReadFileInputStream> input_;
  memory::MemoryPool* const pool_;

 private:
  // 这里返回一个 `SeekableInputStream`, 注意接口上这玩意可能是 Lazy 的,
  // 不过翻了一下, 这个实现是不会返回 lazy 的, 因为内部实现 `readInternal`
  // 没有返回 Lazy.
  std::unique_ptr<SeekableInputStream> readBuffer(
      uint64_t offset,
      uint64_t length) const;

  std::tuple<const char*, uint64_t> readInternal(
      uint64_t offset,
      uint64_t length,
      std::optional<size_t> i = std::nullopt) const;

  void readToBuffer(
      uint64_t offset,
      folly::Range<char*> allocated,
      const LogType logType);

  folly::Range<char*> allocate(const velox::common::Region& region) {
    // Save the file offset and the buffer to which we'll read it
    //
    // 根据 region 申请需要的 `offsets_` 和 `buffer_`, 返回一个 `folly::Range`
    // 的 API.
    offsets_.push_back(region.offset);
    buffers_.emplace_back(
        allocPool_->allocateFixed(region.length), region.length);
    return folly::Range<char*>(buffers_.back().data(), region.length);
  }

  bool useVRead() const;
  void sortRegions();
  void mergeRegions();

  // tries and merges WS read regions into one
  bool tryMerge(
      velox::common::Region& first,
      const velox::common::Region& second);

  uint64_t maxMergeDistance_;
  std::optional<bool> wsVRLoad_;
  std::unique_ptr<memory::AllocationPool> allocPool_;

  // Regions enqueued for reading
  std::vector<velox::common::Region> regions_;

  // Offsets in the file to which the corresponding Region belongs
  //
  // 内部 `regions_` 的 offset.
  std::vector<uint64_t> offsets_;

  // Buffers allocated for reading each Region.
  //
  // 需要读 `regions_` 的 buffer.
  std::vector<folly::Range<char*>> buffers_;

  // Maps the position in which the Region was originally enqueued to the
  // position that it went to after sorting and merging. Thus this maps from the
  // enqueued position to its corresponding buffer offset.
  std::vector<size_t> enqueuedToBufferOffset_;
};

} // namespace facebook::velox::dwio::common
