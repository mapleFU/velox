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

#include "velox/expression/Expr.h"

namespace facebook::velox::exec {

/// Constant, Cast, Coalesce, Conjunct(And, Or), FieldReference, Switch, Lambda, Try.
class SpecialForm : public Expr {
 public:
  SpecialForm(
      TypePtr type,
      std::vector<ExprPtr> inputs,
      const std::string& name,
      bool supportsFlatNoNullsFastPath,
      bool trackCpuUsage)
      : Expr(
            std::move(type),
            std::move(inputs),
            name,
            true /* specialForm */,
            supportsFlatNoNullsFastPath,
            trackCpuUsage) {}

  // This is safe to call only after all metadata is computed for input
  // expressions.
  //
  // Q: 我也想知道这个为啥需要有一个特殊的 PropagateNulls 的过程.
  // A: 这个接口的引入在: https://github.com/facebookincubator/velox/pull/5287
  //
  // 1. 对于 VectorFunction 的表达式, 从 VectorFuction 上直接拿到 propagateNull 就可以了.
  //    Expr 上也有 `bool propagateNulls() const` 的接口.
  // 2. 对于 SpecialForm, 额外从 `!vectorFunction` 来计算 propagateNulls.
  //
  // 其实本质上就是说, Expr 也是一种特殊的 SpecialFunction, 也需要 computePropagatesNulls.
  virtual void computePropagatesNulls() {
    VELOX_NYI();
  }
};
} // namespace facebook::velox::exec
