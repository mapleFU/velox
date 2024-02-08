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

#include "velox/expression/PeeledEncoding.h"

#include "velox/expression/EvalCtx.h"

namespace facebook::velox::exec {

/*static*/ std::shared_ptr<PeeledEncoding> PeeledEncoding::peel(
    const std::vector<VectorPtr>& vectorsToPeel,
    const SelectivityVector& rows,
    DecodedVector& decodedVector,
    bool canPeelsHaveNulls,
    std::vector<VectorPtr>& peeledVectors) {
  std::shared_ptr<PeeledEncoding> peeledEncoding(new PeeledEncoding());
  if (peeledEncoding->peelInternal(
          vectorsToPeel,
          rows,
          decodedVector,
          canPeelsHaveNulls,
          peeledVectors)) {
    return peeledEncoding;
  }
  return nullptr;
}

/*static*/ std::shared_ptr<PeeledEncoding> PeeledEncoding::peel(
    const std::vector<VectorPtr>& vectorsToPeel,
    const SelectivityVector& rows,
    LocalDecodedVector& decodedVector,
    bool canPeelsHaveNulls,
    std::vector<VectorPtr>& peeledVectors) {
  return peel(
      vectorsToPeel,
      rows,
      *decodedVector.get(),
      canPeelsHaveNulls,
      peeledVectors);
}

SelectivityVector* PeeledEncoding::translateToInnerRows(
    const SelectivityVector& outerRows,
    LocalSelectivityVector& innerRowsHolder) const {
  VELOX_CHECK(wrapEncoding_ != VectorEncoding::Simple::FLAT);
  if (wrapEncoding_ == VectorEncoding::Simple::CONSTANT) {
    // Constant: 直接生成长度 `constantWrapIndex_ + 1`, 只有 `[constantWrapIndex_]` 合法的 innerRows.
    auto newRows = innerRowsHolder.get(constantWrapIndex_ + 1, false);
    newRows->setValid(constantWrapIndex_, true);
    newRows->updateBounds();
    return newRows;
  }
  auto baseSize = baseSize_;
  auto indices = wrap_->as<vector_size_t>();
  // If the wrappers add nulls, do not enable the inner rows. The
  // indices for places that a dictionary sets to null are not
  // defined. Null adding dictionaries are not peeled off non
  // null-propagating Exprs.
  auto flatNulls = wrapNulls_ ? wrapNulls_->as<uint64_t>() : nullptr;

  auto* newRows = innerRowsHolder.get(baseSize, false);
  velox::translateToInnerRows(outerRows, indices, flatNulls, *newRows);

  return newRows;
}

namespace {
inline void setPeeled(
    const VectorPtr& leaf,
    int32_t fieldIndex,
    std::vector<VectorPtr>& peeled) {
  VELOX_DCHECK(peeled.size() > fieldIndex);
  peeled[fieldIndex] = leaf;
}

/// Returns true if 'wrapper' is a dictionary vector over a flat vector.
bool isDictionaryOverFlat(const BaseVector& wrapper) {
  return wrapper.encoding() == VectorEncoding::Simple::DICTIONARY &&
      wrapper.valueVector()->isFlatEncoding();
}
} // namespace

void PeeledEncoding::setDictionaryWrapping(
    DecodedVector& decoded,
    const SelectivityVector& rows,
    BaseVector& firstWrapper) {
  wrapEncoding_ = VectorEncoding::Simple::DICTIONARY;
  baseSize_ = decoded.base()->size();
  // 如果是一层的话, 直接返回
  // 否则用 decoded.dictionaryWrapping 来把中间层缝合(使用 DecodedVector 来缝合).
  if (isDictionaryOverFlat(firstWrapper)) {
    // Re-use indices and nulls buffers.
    wrap_ = firstWrapper.wrapInfo();
    wrapNulls_ = firstWrapper.nulls();
    return;
  }
  auto wrapping = decoded.dictionaryWrapping(firstWrapper, rows.end());
  wrap_ = std::move(wrapping.indices);
  wrapNulls_ = std::move(wrapping.nulls);
}

bool PeeledEncoding::peelInternal(
    const std::vector<VectorPtr>& vectorsToPeel,
    const SelectivityVector& rows,
    DecodedVector& decodedVector,
    bool canPeelsHaveNulls,
    std::vector<VectorPtr>& peeledVectors) {
  auto numFields = vectorsToPeel.size();
  std::vector<VectorPtr> maybePeeled;
  std::vector<bool> constantFields;
  int numLevels = 0;
  bool peeled;
  bool nonConstant = false;
  int32_t firstPeeled = -1;
  // 尝试匹配 `numLevels` 层, 前面的东西相当于已经匹配, 每个循环会扫一遍整个 `numFields`.
  // peeledVectors 会在每次循环中被更新, 相当于上一轮的 Peel 结果.
  //
  // 每轮如果整轮 Peel 成功, 会增加 numLevels, 然后推进下一轮. Peeled 表示本轮是否成功,
  // nonConstant 表示本轮是否有非 constant 的 vector.
  //
  // 想了一下, 这里有一层非常关键的特性, 就是比如 DICT(DICT(...)), 如果能 Peel 出最内层的
  // Dict, 那么可以只用一层的 Dict 来表示, 外层的 indices 都是没必要的. 所以这里每次一层检查
  // 完, 如果 Peel 成功, 就把 Peel 出来的 vector 保存下来到 `peeledVectors`, 然后继续下一轮.
  do {
    peeled = true;
    // 本轮尝试的 indices.
    BufferPtr firstIndices;
    maybePeeled.resize(numFields);
    for (int fieldIndex = 0; fieldIndex < numFields; fieldIndex++) {
      // 这里应该对应 `peeledVectors = std::move(maybePeeled);` 这里的情况,
      // 即第一次用 `vectorsToPeel`, 之后用 `peeledVectors`.
      auto leaf = peeledVectors.empty() ? vectorsToPeel[fieldIndex]
                                        : peeledVectors[fieldIndex];
      if (leaf == nullptr) {
        continue;
      }
      if (leaf->isLazy() && leaf->asUnchecked<LazyVector>()->isLoaded()) {
        auto lazy = leaf->asUnchecked<LazyVector>();
        // 拿到之前 load 过的 vector.
        leaf = lazy->loadedVectorShared();
      }
      // constant 表示上一层这里是 constant. 上轮是 Constant 的话本轮一定是.
      if (!constantFields.empty() && constantFields[fieldIndex]) {
        setPeeled(leaf, fieldIndex, maybePeeled);
        continue;
      }
      // TODO: consider removing check for numLevels by ensuring this will not
      // affect CSE.
      // Context: numLevels == 0 was done in peelEncodings but not in
      // applyFunctionWithPeeling. This check prevents peeling if other
      // dictionary vectors have further layers. For eg. take D1(D2(FLAT)),
      // D1(C1), if this check is in place then peeling will stop at D1,
      // otherwise peeling will continue and peel off D2 as well.
      //
      // 这个地方的逻辑应该只处理了一层的 Constant. 按上面注释所说，多层就不考虑 Peeling 
      // Constant 了.
      if (numLevels == 0 && leaf->isConstantEncoding()) {
        // 设置 constant flag, 然后拷贝 到 maybePeeled, 继续下一轮.
        setPeeled(leaf, fieldIndex, maybePeeled);
        constantFields.resize(numFields);
        constantFields.at(fieldIndex) = true;
        continue;
      }
      nonConstant = true;
      auto encoding = leaf->encoding();
      if (encoding == VectorEncoding::Simple::DICTIONARY) {
        if (!canPeelsHaveNulls && leaf->rawNulls()) {
          // A dictionary that adds nulls over an Expr that is not null for a
          // null argument cannot be peeled.
          //
          // 有 Null 并设置了 canPeelsHaveNulls 为 false, 则不能 peel.
          peeled = false;
          break;
        }
        // 拿到 indices.
        BufferPtr indices = leaf->wrapInfo();
        if (!firstIndices) {
          firstIndices = std::move(indices);
        } else if (indices != firstIndices) {
          // different fields use different dictionaries
          //
          // NOTE(mwish): 这个地方是不是只能证明 indices 不相同, 本层的 indices 不相同,
          //  上层可能指向同一个 dictionary.
          peeled = false;
          break;
        }
        if (firstPeeled == -1) {
          firstPeeled = fieldIndex;
        }
        // 这里设置的是 valueVector, 这里每次剥离出来的都是 value.
        setPeeled(leaf->valueVector(), fieldIndex, maybePeeled);
      } else {
        // Non-peelable encoding.
        peeled = false;
        break;
      }
    }
    // 本层是 Peeled, 尝试设置 maybePeeled 和推进 Level.
    if (peeled) {
      ++numLevels;
      peeledVectors = std::move(maybePeeled);
    }
  } while (peeled && nonConstant);

  if (numLevels == 0 && nonConstant) {
    return false;
  }

  if (firstPeeled == -1) {
    // 最后一层遇到的都是 constant, 没有 dictionary.
    wrapEncoding_ = VectorEncoding::Simple::CONSTANT;
    // Check if constant encoding can be peeled off too if the input is of the
    // form Constant(complex).
    if (peeledVectors.size() == 1 &&
        peeledVectors.back()->valueVector() != nullptr) {
      auto constVector = peeledVectors.back();
      constantWrapIndex_ = constVector->wrappedIndex(rows.begin());
      peeledVectors[0] = constVector->valueVector();
    } else {
      constantWrapIndex_ = rows.begin();
    }
  } else {
    auto firstWrapper = vectorsToPeel[firstPeeled];
    // Check if constant encoding can be peeled off too if the input is of the
    // form Dictionary(Constant(complex)).
    if (peeledVectors.size() == 1 &&
        peeledVectors.back()->isConstantEncoding() &&
        peeledVectors.back()->valueVector() != nullptr) {
      numLevels++; // include the constant layer while decoding.
      peeledVectors.back() = peeledVectors.back()->valueVector();
    }
    // 折叠字典.
    decodedVector.makeIndices(*firstWrapper, rows, numLevels);
    if (decodedVector.isConstantMapping()) {
      // This can only happen if the attempt to peel a constant encoding layer
      // exposed a null complex constant as the base.
      VELOX_CHECK(peeledVectors.size() == 1);
      auto innerIdx = decodedVector.index(rows.begin());
      VELOX_CHECK(peeledVectors.back()->isNullAt(innerIdx));
      wrapEncoding_ = VectorEncoding::Simple::CONSTANT;
      constantWrapIndex_ = innerIdx;
    } else {
      setDictionaryWrapping(decodedVector, rows, *firstWrapper);
      // Make sure all the constant vectors have at least the same length as the
      // base vector after peeling. This will make sure any translated rows
      // point to valid rows in the constant vector.
      if (baseSize_ > rows.end() && !constantFields.empty()) {
        for (int i = 0; i < constantFields.size(); ++i) {
          if (constantFields[i]) {
            peeledVectors[i] =
                BaseVector::wrapInConstant(baseSize_, 0, peeledVectors[i]);
          }
        }
      }
    }
  }
  return true;
}

VectorEncoding::Simple PeeledEncoding::wrapEncoding() const {
  return wrapEncoding_;
}

VectorPtr PeeledEncoding::wrap(
    const TypePtr& outputType,
    velox::memory::MemoryPool* pool,
    VectorPtr peeledResult,
    const SelectivityVector& rows) const {
  VELOX_CHECK(wrapEncoding_ != VectorEncoding::Simple::FLAT);
  VectorPtr wrappedResult;
  if (wrapEncoding_ == VectorEncoding::Simple::DICTIONARY) {
    if (!peeledResult) {
      // If all rows are null, make a constant null vector of the right type.
      wrappedResult =
          BaseVector::createNullConstant(outputType, rows.size(), pool);
    } else {
      BufferPtr nulls;
      if (!rows.isAllSelected()) {
        // The new base vector may be shorter than the original base vector
        // (e.g. if positions at the end of the original vector were not
        // selected for evaluation). In this case some original indices
        // corresponding to non-selected rows may point past the end of the base
        // vector. Disable these by setting corresponding positions to null.
        nulls = AlignedBuffer::allocate<bool>(rows.size(), pool, bits::kNull);
        // Set the active rows to non-null.
        rows.clearNulls(nulls);
        if (wrapNulls_) {
          // Add the nulls from the wrapping.
          bits::andBits(
              nulls->asMutable<uint64_t>(),
              wrapNulls_->as<uint64_t>(),
              rows.begin(),
              rows.end());
        }
        // Reset nulls buffer if all positions happen to be non-null.
        if (bits::isAllSet(
                nulls->as<uint64_t>(), 0, rows.end(), bits::kNotNull)) {
          nulls.reset();
        }
      } else {
        nulls = wrapNulls_;
      }
      wrappedResult = BaseVector::wrapInDictionary(
          std::move(nulls), wrap_, rows.end(), std::move(peeledResult));
    }
  } else {
    wrappedResult = BaseVector::wrapInConstant(
        rows.size(), constantWrapIndex_, std::move(peeledResult));
  }
  return wrappedResult;
}
} // namespace facebook::velox::exec
