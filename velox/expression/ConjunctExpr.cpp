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
#include "velox/expression/ConjunctExpr.h"
#include "velox/expression/BooleanMix.h"
#include "velox/expression/FieldReference.h"
#include "velox/expression/ScopedVarSetter.h"

namespace facebook::velox::exec {

namespace {

// 返回本轮的 errorMask, 本轮没有就返回 Null, 
// 然后给 context 里面的 errors 塞最终的 error.
uint64_t* rowsWithError(
    const SelectivityVector& rows,
    const SelectivityVector& activeRows,
    EvalCtx& context,
    ErrorVectorPtr& previousErrors,
    LocalSelectivityVector& errorRowsHolder) {
  auto errors = context.errors();
  if (!errors) {
    // No new errors. Put the old errors back.
    context.swapErrors(previousErrors);
    return nullptr;
  }
  // 下面会从 `errorRowsHolder` 来 Normalize 本轮的 errorMask.
  uint64_t* errorMask = nullptr;
  SelectivityVector* errorRows = errorRowsHolder.get();
  if (!errorRows) {
    errorRows = errorRowsHolder.get(rows.end(), false);
  }
  errorMask = errorRows->asMutableRange().bits();
  std::fill(errorMask, errorMask + bits::nwords(rows.end()), 0);
  // A 1 in activeRows with a not null in errors null flags makes a 1
  // in errorMask.
  bits::andBits(
      errorMask,
      activeRows.asRange().bits(),
      errors->rawNulls(),
      rows.begin(),
      std::min(errors->size(), rows.end()));
  if (previousErrors) {
    // Add the new errors to the previous ones and free the new errors.
    // 和之前的 error 做 merge, 这里就直接把新的 error 塞到了 previousErrors 里面.
    bits::forEachSetBit(
        errors->rawNulls(), rows.begin(), errors->size(), [&](int32_t row) {
          context.addError(
              row,
              *std::static_pointer_cast<std::exception_ptr>(
                  errors->valueAt(row)),
              previousErrors);
        });
    context.swapErrors(previousErrors);
    previousErrors = nullptr;
  }
  return errorMask;
}

void finalizeErrors(
    const SelectivityVector& rows,
    const SelectivityVector& activeRows,
    bool throwOnError,
    EvalCtx& context) {
  auto errors = context.errors();
  if (!errors) {
    return;
  }
  // Pre-existing errors outside of initial active rows are preserved. Errors in
  // the initial active rows but not in the final active rows are cleared.
  auto size =
      std::min(std::min(rows.size(), activeRows.size()), errors->size());
  for (auto i = 0; i < size; ++i) {
    if (errors->isNullAt(i)) {
      continue;
    }
    // rows.isValid && !activeRows.isValid 说明遇到了最高级别优先级, 即 AND 的 false,
    // OR 的 true. 这种情况下, 把这个位置的 error 设置为 Null.
    if (rows.isValid(i) && !activeRows.isValid(i)) {
      errors->setNull(i, true);
    }
    // 如果 throwOnError 为 true, 并且这个位置有 error, 就抛出这个 error.
    if (throwOnError && !errors->isNullAt(i)) {
      auto exceptionPtr =
          std::static_pointer_cast<std::exception_ptr>(errors->valueAt(i));
      std::rethrow_exception(*exceptionPtr);
    }
  }
}
} // namespace

void ConjunctExpr::evalSpecialForm(
    const SelectivityVector& rows,
    EvalCtx& context,
    VectorPtr& result) {
  /*
  参考: https://facebookincubator.github.io/velox/develop/expression-evaluation.html#error-handling-in-and-or-try
  对 AND 而言, false > error > true; 对 OR 而言, true > error > false.

  When evaluating AND expression, If some input generates an error for a row on which 
  another input returns false,  the error is suppressed.

  When evaluating OR expression, If some input generates an error for a row on which
  another input returns true, the error is suppressed.

  The error suppression logic in the AND and OR expressions is designed to deliver 
  consistent results regardless of the order of the conjuncts.
  */
  // TODO Revisit error handling
  bool throwOnError = *context.mutableThrowOnError();
  ScopedVarSetter saveError(context.mutableThrowOnError(), false);
  context.ensureWritable(rows, type(), result);
  auto flatResult = result->asFlatVector<bool>();
  // clear nulls from the result for the active rows.
  if (flatResult->mayHaveNulls()) {
    auto& nulls = flatResult->mutableNulls(rows.end());
    rows.clearNulls(nulls);
  }
  // Initialize result to all true for AND and all false for OR.
  auto values = flatResult->mutableValues(rows.end())->asMutable<uint64_t>();
  if (isAnd_) {
    // 对 AND, 先给所有的 `rows` 设置 True.
    bits::orBits(values, rows.asRange().bits(), rows.begin(), rows.end());
  } else {
    // 对 OR, &(!rows), 应该相当于给 `rows` 设置 False.
    bits::andWithNegatedBits(
        values, rows.asRange().bits(), rows.begin(), rows.end());
  }

  // 如果是 OR, 设置为 !finalSelection.
  // TODO(mwish): 为什么 AND 不要呢?
  // OR: fix finalSelection at "rows" unless already fixed
  ScopedFinalSelectionSetter scopedFinalSelectionSetter(
      context, &rows, !isAnd_);

  bool handleErrors = false;
  LocalSelectivityVector errorRows(context);
  // 先选全 rows 作为 activeRows, 如果拿到 AND 的 false, OR 的 true 这几个
  // 最高优先级的对象, 后面就可以不 eval 了.
  LocalSelectivityVector activeRowsHolder(context, rows);
  auto activeRows = activeRowsHolder.get();
  VELOX_DCHECK(activeRows != nullptr);
  int32_t numActive = activeRows->countSelected();
  for (int32_t i = 0; i < inputs_.size(); ++i) {
    VectorPtr inputResult;
    VectorRecycler inputResultRecycler(inputResult, context.vectorPool());
    // 第一次 handleErrors 为 false, 不会执行这个分支.
    // 后续有 error 的时候, 会执行这个分支, 把 ctx 中 err 拿出来.
    ErrorVectorPtr errors;
    if (handleErrors) {
      context.swapErrors(errors);
    }

    SelectivityTimer timer(selectivity_[inputOrder_[i]], numActive);
    // https://github.com/facebookincubator/velox/pull/7433
    // 对于 AND 这样的, 尝试 load. 这段代码和上面的 pr 有关 (即 Expr 框架中没有 load).
    if (evaluatesArgumentsOnNonIncreasingSelection()) {
      // Exclude loading rows that we know for sure will have a false result.
      for (auto* field : inputs_[inputOrder_[i]]->distinctFields()) {
        if (multiplyReferencedFields_.count(field) > 0) {
          context.ensureFieldLoaded(field->index(context), *activeRows);
        }
      }
    }
    inputs_[inputOrder_[i]]->eval(*activeRows, context, inputResult);
    if (context.errors()) {
      handleErrors = true;
    }
    // extraActive: active 中新产生的 errors.
    uint64_t* extraActive = nullptr;
    if (handleErrors) {
      // Add rows with new errors to activeRows and merge these with
      // previous errors.
      extraActive =
          rowsWithError(rows, *activeRows, context, errors, errorRows);
    }
    updateResult(inputResult.get(), context, flatResult, activeRows);
    if (extraActive) {
      uint64_t* activeBits = activeRows->asMutableRange().bits();
      bits::orBits(activeBits, extraActive, rows.begin(), rows.end());
      activeRows->updateBounds();
    }
    numActive = activeRows->countSelected();
    selectivity_[inputOrder_[i]].addOutput(numActive);

    if (!numActive) {
      break;
    }
  }
  // Clear errors for 'rows' that are not in 'activeRows'.
  // 这里 throwOnError 在执行期间是 false, 因为有一个 False/True 可能覆盖 Error 的逻辑,
  // 然后在收尾的时候处理异常.
  finalizeErrors(rows, *activeRows, throwOnError, context);
  if (!reorderEnabledChecked_) {
    reorderEnabled_ = context.execCtx()
                          ->queryCtx()
                          ->queryConfig()
                          .adaptiveFilterReorderingEnabled();
    reorderEnabledChecked_ = true;
  }
  if (reorderEnabled_) {
    // 在执行完一轮以后, 尝试 reorder.
    maybeReorderInputs();
  }
}

void ConjunctExpr::maybeReorderInputs() {
  // 根据 timeToDropValue() 来 reorder inputs.
  // 这里 timeToDropValue() 越小, 说明越快 drop. 如果啥都没 drop,
  // 会返回总体执行的 Time.
  bool reorder = false;
  for (auto i = 1; i < inputs_.size(); ++i) {
    if (selectivity_[inputOrder_[i - 1]].timeToDropValue() >
        selectivity_[inputOrder_[i]].timeToDropValue()) {
      reorder = true;
      break;
    }
  }
  if (reorder) {
    std::sort(
        inputOrder_.begin(),
        inputOrder_.end(),
        [this](size_t left, size_t right) {
          return selectivity_[left].timeToDropValue() <
              selectivity_[right].timeToDropValue();
        });
  }
}

namespace {
// helper functions for conjuncts operating on values, nulls and active rows a
// word at a time.
inline void setFalseForOne(uint64_t active, uint64_t source, uint64_t& target) {
  target &= ~active | ~source;
}

inline void setTrueForOne(uint64_t active, uint64_t source, uint64_t& target) {
  target |= active & source;
}

inline void
setPresentForOne(uint64_t active, uint64_t source, uint64_t& target) {
  target |= active & source;
}

inline void
setNonPresentForOne(uint64_t active, uint64_t source, uint64_t& target) {
  target &= ~active | ~source;
}

/// Update 函数:
/// - value: 值
/// - present: Null/NonNull
/// - active: 当前 activeRows
/// 用来更新的:
/// - testValue: 本轮算出来的 value
/// - testPresent: 本轮算出来的 null


inline void updateAnd(
    uint64_t& resultValue,
    uint64_t& resultPresent,
    uint64_t& active,
    uint64_t testValue,
    uint64_t testPresent) {
  // testFalse: 新增的 non-null False
  auto testFalse = ~testValue & testPresent;
  // resultValue 中 testFalse 的地方设置为 False.
  // (因为对于 AND, false > null, 所以有 Null 也设置成 false, 没管 Present)
  setFalseForOne(active, testFalse, resultValue);
  // resultPresent 中 testFalse 的地方设置为 Present.
  // (因为对于 AND, false > null, 所以这里设置为 Present.)
  setPresentForOne(active, testFalse, resultPresent);
  // 拿到更新 resultPresent 之后的 resultTrue.
  auto resultTrue = resultValue & resultPresent;
  // 把 resultValue 中在 testPresent 为 Null 的地方设置为 Null.
  setNonPresentForOne(
      active, resultPresent & resultTrue & ~testPresent, resultPresent);
  // 取消新增 non-null false 的 active.
  active &= ~testFalse;
}

inline void updateOr(
    uint64_t& resultValue,
    uint64_t& resultPresent,
    uint64_t& active,
    uint64_t testValue,
    uint64_t testPresent) {
  auto testTrue = testValue & testPresent;
  setTrueForOne(active, testTrue, resultValue);
  setPresentForOne(active, testTrue, resultPresent);
  auto resultFalse = ~resultValue & resultPresent;
  setNonPresentForOne(
      active, resultPresent & resultFalse & ~testPresent, resultPresent);
  active &= ~testTrue;
}

} // namespace

// 在初始化的时候, AND 的时候会初始化为 true, OR 的时候会初始化为 false.
// 所以, and 只关注 NULL 和 FALSE, or 只关注 NULL 和 TRUE.
void ConjunctExpr::updateResult(
    BaseVector* inputResult,
    EvalCtx& context,
    FlatVector<bool>* result,
    SelectivityVector* activeRows) {
  // Set result and clear active rows for the positions that are decided.
  // 这两个用 `uint64_t` 因为后续 `bits::forEachWord` 是按 word 来处理的.
  const uint64_t* values = nullptr;
  const uint64_t* nulls = nullptr;
  switch (getFlatBool(
      inputResult,
      *activeRows,
      context,
      &tempValues_,
      &tempNulls_,
      false,
      &values,
      &nulls)) {
    case BooleanMix::kAllNull:
      // 全 Null 去 addNulls.
      result->addNulls(*activeRows);
      return;
    case BooleanMix::kAllFalse:
      // AND 全是 False 的时候, 直接给 activeRows 设置 False.
      if (isAnd_) {
        // Clear the nulls and values for all active rows.
        if (result->mayHaveNulls()) {
          // And 中, false > null.
          activeRows->clearNulls(result->mutableRawNulls());
        }
        // 手动只 &(!activeRows), 即给 activeRows 设置 False.
        // 这样看 activeRows 应该强制要求没 row select 的地方是 1...
        bits::andWithNegatedBits(
            result->mutableRawValues<uint64_t>(),
            activeRows->asRange().bits(),
            activeRows->begin(),
            activeRows->end());
        activeRows->clearAll();
      }
      return;
    case BooleanMix::kAllTrue:
      if (!isAnd_) {
        if (result->mayHaveNulls()) {
          bits::orBits(
              result->mutableRawNulls(),
              activeRows->asRange().bits(),
              activeRows->begin(),
              activeRows->end());
        }
        bits::orBits(
            result->mutableRawValues<uint64_t>(),
            activeRows->asRange().bits(),
            activeRows->begin(),
            activeRows->end());

        activeRows->clearAll();
      }
      return;
    default: {
      // 非全 True / False / NULL, Fallback 回来处理.
      // 这套代码在这个 patch 做了很多操作优化: https://github.com/facebookincubator/velox/pull/7062
      // 随便看看吧
      uint64_t* resultValues = result->mutableRawValues<uint64_t>();
      uint64_t* resultNulls = nullptr;
      if (nulls || result->mayHaveNulls()) {
        resultNulls = result->mutableRawNulls();
      }
      auto* activeBits = activeRows->asMutableRange().bits();
      if (isAnd_) {
        bits::forEachWord(
            activeRows->begin(),
            activeRows->end(),
            [&](int32_t index, uint64_t mask) {
              uint64_t nullWord =
                  resultNulls ? resultNulls[index] : bits::kNotNull64;
              uint64_t activeWord = activeBits[index] & mask;
              updateAnd(
                  resultValues[index],
                  nullWord,
                  activeWord,
                  values[index],
                  nulls ? nulls[index] : bits::kNotNull64);
              if (resultNulls) {
                resultNulls[index] = nullWord;
              }
              activeBits[index] &= ~mask | activeWord;
            },
            [&](int32_t index) {
              uint64_t nullWord =
                  resultNulls ? resultNulls[index] : bits::kNotNull64;
              updateAnd(
                  resultValues[index],
                  nullWord,
                  activeBits[index],
                  values[index],
                  nulls ? nulls[index] : bits::kNotNull64);
              if (resultNulls) {
                resultNulls[index] = nullWord;
              }
            });
      } else {
        bits::forEachWord(
            activeRows->begin(),
            activeRows->end(),
            [&](int32_t index, uint64_t mask) {
              uint64_t nullWord =
                  resultNulls ? resultNulls[index] : bits::kNotNull64;
              uint64_t activeWord = activeBits[index] & mask;
              updateOr(
                  resultValues[index],
                  nullWord,
                  activeWord,
                  values[index],
                  nulls ? nulls[index] : bits::kNotNull64);
              if (resultNulls) {
                resultNulls[index] = nullWord;
              }
              activeBits[index] &= ~mask | activeWord;
            },
            [&](int32_t index) {
              uint64_t nullWord =
                  resultNulls ? resultNulls[index] : bits::kNotNull64;
              updateOr(
                  resultValues[index],
                  nullWord,
                  activeBits[index],
                  values[index],
                  nulls ? nulls[index] : bits::kNotNull64);
              if (resultNulls) {
                resultNulls[index] = nullWord;
              }
            });
      }
      activeRows->updateBounds();
    }
  }
}

std::string ConjunctExpr::toSql(
    std::vector<VectorPtr>* complexConstants) const {
  std::stringstream out;
  out << "(" << inputs_[0]->toSql(complexConstants) << ")";
  for (auto i = 1; i < inputs_.size(); ++i) {
    out << " " << name_ << " "
        << "(" << inputs_[i]->toSql(complexConstants) << ")";
  }
  return out.str();
}

// static
TypePtr ConjunctExpr::resolveType(const std::vector<TypePtr>& argTypes) {
  VELOX_CHECK_GT(
      argTypes.size(),
      0,
      "Conjunct expressions expect at least one argument, received: {}",
      argTypes.size());

  for (const auto& argType : argTypes) {
    VELOX_CHECK(
        argType->kind() == TypeKind::BOOLEAN ||
            argType->kind() == TypeKind::UNKNOWN,
        "Conjunct expressions expect BOOLEAN or UNKNOWN arguments, received: {}",
        argType->toString());
  }

  return BOOLEAN();
}

TypePtr ConjunctCallToSpecialForm::resolveType(
    const std::vector<TypePtr>& argTypes) {
  return ConjunctExpr::resolveType(argTypes);
}

ExprPtr ConjunctCallToSpecialForm::constructSpecialForm(
    const TypePtr& type,
    std::vector<ExprPtr>&& compiledChildren,
    bool /* trackCpuUsage */,
    const core::QueryConfig& /*config*/) {
  bool inputsSupportFlatNoNullsFastPath =
      Expr::allSupportFlatNoNullsFastPath(compiledChildren);

  return std::make_shared<ConjunctExpr>(
      type,
      std::move(compiledChildren),
      isAnd_,
      inputsSupportFlatNoNullsFastPath);
}
} // namespace facebook::velox::exec
