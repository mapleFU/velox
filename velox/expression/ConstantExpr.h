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

#include "velox/expression/SpecialForm.h"

namespace facebook::velox::exec {

/// Constant, 这里应该可以利用的是 `value` 的第 0 行, 也可以表示 null.
/// (感觉主要是它这里没有 Scalar 类型, 对于 Scalar 来说, 它懒得搞一套额外的内存管理机制? 或者也是这样能表达
///  Typed Null Value).
///
/// 如果 Constant 是 Null 就不能有 flatNotNullsFastPath(废话你自己都 null 了 fast 个 j)
class ConstantExpr : public SpecialForm {
 public:
  explicit ConstantExpr(VectorPtr value)
      : SpecialForm(
            value->type(),
            std::vector<ExprPtr>(),
            "literal",
            !value->isNullAt(0) /* supportsFlatNoNullsFastPath */,
            false /* trackCpuUsage */),
        needToSetIsAscii_{value->type()->isVarchar()} {
    VELOX_CHECK_EQ(value->encoding(), VectorEncoding::Simple::CONSTANT);
    // sharedConstantValue_ may be modified so we should take our own copy to
    // prevent sharing across threads.
    sharedConstantValue_ =
        BaseVector::wrapInConstant(1, 0, std::move(value), true);
  }

  void evalSpecialForm(
      const SelectivityVector& rows,
      EvalCtx& context,
      VectorPtr& result) override;

  void evalSpecialFormSimplified(
      const SelectivityVector& rows,
      EvalCtx& context,
      VectorPtr& result) override;

  const VectorPtr& value() const {
    return sharedConstantValue_;
  }

  VectorPtr& mutableValue() {
    return sharedConstantValue_;
  }

  std::string toString(bool recursive = true) const override;

  std::string toSql(
      std::vector<VectorPtr>* complexConstants = nullptr) const override;

 private:
  // 内部生成的 Constant column
  // 因为这个 constant value 构造来自于 VectorPtr value, 所以会在这上面包一层 Constant.
  VectorPtr sharedConstantValue_;
  // 需要设置 isAscii 的标志位.
  bool needToSetIsAscii_;
};
} // namespace facebook::velox::exec
