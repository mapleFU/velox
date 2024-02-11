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

#include "velox/expression/FieldReference.h"

#include "velox/expression/PeeledEncoding.h"

namespace facebook::velox::exec {

void FieldReference::computeDistinctFields() {
  SpecialForm::computeDistinctFields();
  // 对 FieldReference, distinctFields 来自:
  // 1. 如果 inputs_ 为空, 那感觉就等价于父级的表达式( `Expr::computeDistinctFields()` 会计算 input ).
  // 2. 否则其实自身没有 distinctFields_ 和 multiplyReferencedFields_, 会把自己作为 underlying 加入进去.
  if (inputs_.empty()) {
    mergeFields(
        distinctFields_,
        multiplyReferencedFields_,
        {this->as<FieldReference>()});
  }
}

void FieldReference::apply(
    const SelectivityVector& rows,
    EvalCtx& context,
    VectorPtr& result) {
  const RowVector* row;
  DecodedVector decoded;
  VectorPtr input;
  std::shared_ptr<PeeledEncoding> peeledEncoding;
  bool useDecode = false;
  LocalSelectivityVector nonNullRowsHolder(*context.execCtx());
  const SelectivityVector* nonNullRows = &rows;
  // 抽出 Input Rows 到 `row` 中.
  if (inputs_.empty()) {
    // 不含 Input 就是直接从 EvalCtx 中取出来(从 Input Vector 中抽取)
    row = context.row();
  } else {
    // 执行第一个表达式(也只有一个), 从中执行并且抽取.
    // rows 作为完整的限定.
    inputs_[0]->eval(rows, context, input);

    if (auto rowTry = input->as<RowVector>()) {
      // Make sure output is not copied
      if (rowTry->isCodegenOutput()) {
        auto rowType = dynamic_cast<const RowType*>(rowTry->type().get());
        index_ = rowType->getChildIdx(field_);
        result = std::move(rowTry->childAt(index_));
        VELOX_CHECK(result.unique());
        return;
      }
    }

    // 按照 selectivityVector, 丢到 decoded 中.
    decoded.decode(*input, rows);
    if (decoded.mayHaveNulls()) {
      nonNullRowsHolder.get(rows);
      nonNullRowsHolder->deselectNulls(
          decoded.nulls(), rows.begin(), rows.end());
      nonNullRows = nonNullRowsHolder.get();
      if (!nonNullRows->hasSelections()) {
        // 一个都没选中就跑路咯.
        addNulls(rows, decoded.nulls(), context, result);
        return;
      }
    }
    useDecode = !decoded.isIdentityMapping();
    if (useDecode) {
      // 根据 PeeledEncoding 来解码内部.
      // 这里实际是强制用 Peel 来展开, 把乱序的 Dictionary 之类的表达式拍平.
      // 我个人觉得不拍平好一点, 不过这个应该是只有 nested field 才会这样? 感觉其实大部分时候开销并不大
      // (即大部分时候甚至不是表达式的结果吧...是 struct 的话很多时候也是平坦的?)
      std::vector<VectorPtr> peeledVectors;
      LocalDecodedVector localDecoded{context};
      peeledEncoding = PeeledEncoding::peel(
          {input}, *nonNullRows, localDecoded, true, peeledVectors);
      VELOX_CHECK_NOT_NULL(peeledEncoding);
      if (peeledVectors[0]->isLazy()) {
        peeledVectors[0] =
            peeledVectors[0]->as<LazyVector>()->loadedVectorShared();
      }
      VELOX_CHECK(peeledVectors[0]->encoding() == VectorEncoding::Simple::ROW);
      row = peeledVectors[0]->as<const RowVector>();
    } else {
      // `decoded.isIdentityMapping()`, 可以直接用 `input` 的结构.
      VELOX_CHECK(input->encoding() == VectorEncoding::Simple::ROW);
      row = input->as<const RowVector>();
    }
  }
  // 上面的逻辑取出了 input (row), 然后就是根据 index_ 来取出对应的 child.
  if (index_ == -1) {
    auto rowType = dynamic_cast<const RowType*>(row->type().get());
    VELOX_CHECK(rowType);
    index_ = rowType->getChildIdx(field_);
  }
  // 拿到 child, 然后拷贝
  // 如果 unshared 的话, 会要求 load, 不知道为什么, 这个地方是不是可以作为一个优化.
  VectorPtr child =
      inputs_.empty() ? context.getField(index_) : row->childAt(index_);
  if (child->encoding() == VectorEncoding::Simple::LAZY) {
    child = BaseVector::loadedVectorShared(child);
  }
  if (result.get()) {
    if (useDecode) {
      child = peeledEncoding->wrap(type_, context.pool(), child, *nonNullRows);
    }
    result->copy(child.get(), *nonNullRows, nullptr);
  } else {
    // The caller relies on vectors having a meaningful size. If we
    // have a constant that is not wrapped in anything we set its size
    // to correspond to rows.end().
    if (!useDecode && child->isConstantEncoding()) {
      child = BaseVector::wrapInConstant(nonNullRows->end(), 0, child);
    }
    result = useDecode ? std::move(peeledEncoding->wrap(
                             type_, context.pool(), child, *nonNullRows))
                       : std::move(child);
  }
  child.reset();

  // Check for nulls in the input struct. Propagate these nulls to 'result'.
  if (!inputs_.empty() && decoded.mayHaveNulls()) {
    addNulls(rows, decoded.nulls(), context, result);
  }
}

void FieldReference::evalSpecialForm(
    const SelectivityVector& rows,
    EvalCtx& context,
    VectorPtr& result) {
  // 直接产生一个 nullptr 作为 localResult.
  VectorPtr localResult;
  apply(rows, context, localResult);
  // 将 localResult 移动(或者拷贝) 到 result 中.
  context.moveOrCopyResult(localResult, rows, result);
}

void FieldReference::evalSpecialFormSimplified(
    const SelectivityVector& rows,
    EvalCtx& context,
    VectorPtr& result) {
  ExceptionContextSetter exceptionContext(
      {[](VeloxException::Type /*exceptionType*/, auto* expr) {
         return static_cast<Expr*>(expr)->toString();
       },
       this});
  VectorPtr input;
  const RowVector* row;
  if (inputs_.empty()) {
    row = context.row();
  } else {
    VELOX_CHECK_EQ(inputs_.size(), 1);
    inputs_[0]->evalSimplified(rows, context, input);
    BaseVector::flattenVector(input);
    row = input->as<RowVector>();
    VELOX_CHECK(row);
  }
  auto index = row->type()->asRow().getChildIdx(field_);
  if (index_ == -1) {
    index_ = index;
  } else {
    VELOX_CHECK_EQ(index_, index);
  }

  LocalSelectivityVector nonNullRowsHolder(*context.execCtx());
  const SelectivityVector* nonNullRows = &rows;
  if (row->mayHaveNulls()) {
    nonNullRowsHolder.get(rows);
    nonNullRowsHolder->deselectNulls(row->rawNulls(), rows.begin(), rows.end());
    nonNullRows = nonNullRowsHolder.get();
    if (!nonNullRows->hasSelections()) {
      addNulls(rows, row->rawNulls(), context, result);
      return;
    }
  }

  auto& child = row->childAt(index_);
  context.ensureWritable(rows, type_, result);
  result->copy(child.get(), *nonNullRows, nullptr);
  if (row->mayHaveNulls()) {
    addNulls(rows, row->rawNulls(), context, result);
  }
}

std::string FieldReference::toString(bool recursive) const {
  std::stringstream out;
  if (!inputs_.empty() && recursive) {
    appendInputs(out);
    out << ".";
  }
  out << name();
  return out.str();
}

std::string FieldReference::toSql(
    std::vector<VectorPtr>* complexConstants) const {
  std::stringstream out;
  if (!inputs_.empty()) {
    appendInputsSql(out, complexConstants);
    out << ".";
  }
  out << "\"" << name() << "\"";
  return out.str();
}

} // namespace facebook::velox::exec
