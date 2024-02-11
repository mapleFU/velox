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

#include "velox/expression/EvalCtx.h"

namespace facebook::velox::exec {

enum class BooleanMix { kAllTrue, kAllFalse, kAllNull, kMixNonNull, kMix };

// Return a BooleanMix representing the status of boolean values in vector. If
// vector contains a mix of true and false, extract the boolean values to a raw
// buffer valuesOut. valuesOut may point to a raw buffer possessed by vector.
// nullsOut remain unchanged if there is no null in vector. tempValues and
// tempNulls may or may not be set by this function.
//
// 把复杂 Boolean 的状态抽取出来, 并且设置 `mergeNullsToValues`, 如果是的话不会返回 null,
// 而是会把 nulls 合并到 values 里面(if null, then false). 然后返回一个 BooleanMix(整个 vec
// 的状态).
BooleanMix getFlatBool(
    BaseVector* vector,
    const SelectivityVector& activeRows,
    EvalCtx& context,
    BufferPtr* tempValues,
    BufferPtr* tempNulls,
    bool mergeNullsToValues,
    const uint64_t** valuesOut,
    const uint64_t** nullsOut);
} // namespace facebook::velox::exec
