/*
   Licensed to the Apache Software Foundation (ASF) under one
   or more contributor license agreements.  See the NOTICE file
   distributed with this work for additional information
   regarding copyright ownership.  The ASF licenses this file
   to you under the Apache License, Version 2.0 (the
   "License"); you may not use this file except in compliance
   with the License.  You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
 */
#pragma once

#include <onika/cuda/stl_adaptors.h>
#include <onika/flat_tuple.h>

#include <type_traits>
#include <utility>

namespace exaDEM {
/// @brief Tag types + constexpr instances for interaction field names (e.g. `attr::friction`),
/// used with `operator[]` to select a field. Populated by EXADEM_DECLARE_INTERACTION_FIELD.
namespace attr {}

namespace interaction_field_details {
/// @brief Drops the leading `void` placeholder used to sidestep the X-macro trailing-comma
/// problem when building a type list, e.g. drop_first<void, A, B>::type == FlatTuple<A, B>.
template <typename... T>
struct drop_first;
template <typename First, typename... Rest>
struct drop_first<First, Rest...> {
  using type = onika::FlatTuple<Rest...>;
};

/// @brief Returns the I-th element of a FlatTuple regardless of its constness (FlatTuple only
/// exposes a non-const get_nth<I>(), plus a separately-named get_nth_const<I>()).
template <size_t I, typename TupleT>
ONIKA_HOST_DEVICE_FUNC inline decltype(auto) get_nth_any(TupleT& t) {
  if constexpr (std::is_const_v<TupleT>) {
    return t.template get_nth_const<I>();
  } else {
    return t.template get_nth<I>();
  }
}

/// @brief Implementation detail of apply_on_flat_tuple(), expanding the index pack.
template <typename Func, typename TupleT, size_t... Is>
ONIKA_HOST_DEVICE_FUNC inline void apply_on_flat_tuple_impl(Func& func, TupleT& t, std::index_sequence<Is...>) {
  (func(get_nth_any<Is>(t)), ...);
}

/// @brief Host-only twin of apply_on_flat_tuple_impl(), for functors that aren't device-callable
/// (e.g. ClearFunctor/ResizeFunctor, which call std::vector::clear()/resize()).
template <typename Func, typename TupleT, size_t... Is>
inline void cpu_apply_on_flat_tuple_impl(Func& func, TupleT& t, std::index_sequence<Is...>) {
  (func(get_nth_any<Is>(t)), ...);
}

/// @brief Implementation detail of zip_apply_on_flat_tuple(), expanding the index pack.
template <typename Func, typename TupleA, typename TupleB, size_t... Is>
ONIKA_HOST_DEVICE_FUNC inline void zip_apply_on_flat_tuple_impl(Func& func, TupleA& a, TupleB& b,
                                                                std::index_sequence<Is...>) {
  (func(get_nth_any<Is>(a), get_nth_any<Is>(b)), ...);
}
}  // namespace interaction_field_details

/// @brief Applies func(element) to every element of a FlatTuple, in order.
template <typename Func, typename TupleT>
ONIKA_HOST_DEVICE_FUNC inline void apply_on_flat_tuple(Func& func, TupleT& t) {
  interaction_field_details::apply_on_flat_tuple_impl(func, t, std::make_index_sequence<TupleT::size()>{});
}

/// @brief Host-only twin of apply_on_flat_tuple(), for functors that aren't device-callable (see
/// cpu_apply_on_flat_tuple_impl()).
template <typename Func, typename TupleT>
inline void cpu_apply_on_flat_tuple(Func& func, TupleT& t) {
  interaction_field_details::cpu_apply_on_flat_tuple_impl(func, t, std::make_index_sequence<TupleT::size()>{});
}

/// @brief Applies func(a_element, b_element) position-by-position over two FlatTuples built
/// from the same field list (hence the same size and field order).
template <typename Func, typename TupleA, typename TupleB>
ONIKA_HOST_DEVICE_FUNC inline void zip_apply_on_flat_tuple(Func& func, TupleA& a, TupleB& b) {
  static_assert(TupleA::size() == TupleB::size(), "zip_apply_on_flat_tuple: mismatched tuple sizes");
  interaction_field_details::zip_apply_on_flat_tuple_impl(func, a, b, std::make_index_sequence<TupleA::size()>{});
}

/// @brief zip_apply_on_flat_tuple functor: `vec[idx] = value`, indexing via vector_data() (for
/// a per-field CudaMMVector, e.g. ClassifierContainer's `fm_`/`bk_`).
struct StoreAtIndexFunctor {
  size_t idx_;
  template <typename VecT, typename ValueT>
  ONIKA_HOST_DEVICE_FUNC inline void operator()(VecT& vec, const ValueT& value) const {
    onika::cuda::vector_data(vec)[idx_] = value;
  }
};
/// @brief zip_apply_on_flat_tuple functor: `value = vec[idx]`, symmetric to StoreAtIndexFunctor.
struct LoadAtIndexFunctor {
  size_t idx_;
  template <typename ValueT, typename VecT>
  ONIKA_HOST_DEVICE_FUNC inline void operator()(ValueT& value, VecT& vec) const {
    value = onika::cuda::vector_data(vec)[idx_];
  }
};

/// @brief Same as StoreAtIndexFunctor, but indexes via the container's own operator[] instead of
/// vector_data(...): needed for a onika::cuda::span<T> (InteractionWrapper's `fm_`/`bk_`), whose
/// operator[] is device-compatible directly (vector_data has no span overload).
struct StoreAtIndexSpanFunctor {
  size_t idx_;
  template <typename VecT, typename ValueT>
  ONIKA_HOST_DEVICE_FUNC inline void operator()(VecT& vec, const ValueT& value) const {
    vec[idx_] = value;
  }
};

/// @brief Span-safe counterpart of LoadAtIndexFunctor (see StoreAtIndexSpanFunctor for why).
struct LoadAtIndexSpanFunctor {
  size_t idx_;
  template <typename ValueT, typename VecT>
  ONIKA_HOST_DEVICE_FUNC inline void operator()(ValueT& value, VecT& vec) const {
    value = vec[idx_];
  }
};

/// @brief zip_apply_on_flat_tuple functor: `dst = span{vector_data(src), vector_size(src)}`,
/// used by InteractionWrapper<IT>'s constructors to build a span<T> tuple from a
/// ClassifierContainer<IT>'s VectorT<T> tuple (same field list/order).
struct ToSpanFunctor {
  template <typename SpanT, typename VecT>
  inline void operator()(SpanT& dst, VecT& src) const {
    dst = SpanT{onika::cuda::vector_data(src), onika::cuda::vector_size(src)};
  }
};
}  // namespace exaDEM

// ============================================================================================
// DANGER ZONE -- macro territory below.
//
// Everything from here to the end of the file is X-macro / token-pasting machinery that
// generates tags, tuple types, and operator[] overloads for the Interaction/InnerBondInteraction/
// ClassifierContainer<IT>/InteractionWrapper<IT> field system. It is fragile and easy to break in
// ways that are hard to diagnose:
//   - Macros here assume specific member names in the *calling* struct (`fm_` or `bk_`), never
//     passed as an argument, only relied upon by convention -- see EXADEM_INTERACTION_FIELD_ACCESSOR
//     vs EXADEM_INTERACTION_FIELD_ACCESSOR_BK below. Using the wrong one silently targets the
//     wrong tuple member if both happen to exist.
//   - EXADEM_PP_CONCAT needs the two-level indirection (EXADEM_PP_CONCAT_ / EXADEM_PP_CONCAT) to
//     force expansion of its arguments (e.g. a MODE token) before pasting; collapsing it to one
//     level looks equivalent but silently pastes the unexpanded macro name instead.
//   - Field lists (EXADEM_INTERACTION_FIELDS, EXADEM_INNER_BOND_FIELDS, ...) must have their
//     INDEX column match their physical position in the list (0, 1, 2, ...) -- nothing here
//     checks that automatically, a mismatch just returns/stores the wrong field silently at
//     runtime (only out-of-range indices are caught, via the static_asserts below).
//   - NEW_TAG must be used exactly once per field NAME (the actual tag declaration); every other
//     reuse of that NAME across other field lists must say SHARED_TAG, or you get a redefinition
//     error. NEW_TAG/SHARED_TAG is only consulted by EXADEM_INTERACTION_FIELD_TAGS -- every other
//     macro here (tuple-type builders, accessor generators) ignores MODE and regenerates
//     unconditionally for every entry in whatever list it's given.
//   - clang-format has previously mangled some of these multi-line macro definitions (backslash
//     continuations) into nonsense on save; if a macro here looks subtly wrong after a save,
//     check that first before assuming the logic itself is broken.
// If you're not modifying the field-list system itself, you almost certainly want to edit a
// *field list* (e.g. EXADEM_INTERACTION_FIELDS in interaction.hpp) or a *call site* of these
// macros, not the macros themselves. Change things here only with real care, and recheck every
// call site afterwards.
// ============================================================================================

// Two-level token paste, needed so macro arguments (e.g. a MODE token) are fully expanded
// before being pasted.
#define EXADEM_PP_CONCAT_(a, b) a##b
#define EXADEM_PP_CONCAT(a, b) EXADEM_PP_CONCAT_(a, b)

// Declares one field tag `attr::_NAME` / `attr::NAME`. This is the real declaration; it must
// be invoked exactly once per NAME (see the NEW_TAG / SHARED_TAG dispatch below).
#define EXADEM_DECLARE_INTERACTION_FIELD(NAME) \
  namespace exaDEM {                           \
  namespace attr {                             \
  struct _##NAME {};                           \
  static constexpr _##NAME NAME{};             \
  }                                            \
  }

// MODE dispatch used by EXADEM_INTERACTION_FIELD_TAGS: NEW_TAG declares, SHARED_TAG is a no-op.
#define EXADEM_FIELD_MODE_NEW_TAG(TYPE, NAME, INDEX) EXADEM_DECLARE_INTERACTION_FIELD(NAME)
#define EXADEM_FIELD_MODE_SHARED_TAG(TYPE, NAME, INDEX)

#define EXADEM_FIELD_TAG_ITEM(TYPE, NAME, INDEX, MODE) EXADEM_PP_CONCAT(EXADEM_FIELD_MODE_, MODE)(TYPE, NAME, INDEX)

// Declares every NEW_TAG entry from a field list LIST(X), where LIST expands to
// X(TYPE, NAME, INDEX, MODE) once per field; SHARED_TAG entries are skipped. Call at namespace
// scope, once for the whole field list.
#define EXADEM_INTERACTION_FIELD_TAGS(LIST) LIST(EXADEM_FIELD_TAG_ITEM)

// Builds the onika::FlatTuple<...> backing storage type from a field list LIST(X)
// (only TYPE is used; NAME/INDEX/MODE are ignored here). Field order in LIST must match INDEX.
#define EXADEM_FIELD_TYPE_ITEM(TYPE, NAME, INDEX, MODE) , TYPE
#define EXADEM_INTERACTION_TUPLE_TYPE(LIST) \
  ::exaDEM::interaction_field_details::drop_first<void LIST(EXADEM_FIELD_TYPE_ITEM)>::type

// Same as above, but wraps each field's type in VectorT<...> (VectorT must be an alias template
// visible unqualified where this is expanded), for SoA containers holding one vector per field.
#define EXADEM_FIELD_VECTOR_TYPE_ITEM(TYPE, NAME, INDEX, MODE) , VectorT<TYPE>
#define EXADEM_INTERACTION_VECTOR_TUPLE_TYPE(LIST) \
  ::exaDEM::interaction_field_details::drop_first<void LIST(EXADEM_FIELD_VECTOR_TYPE_ITEM)>::type

// operator[](attr::_NAME) overloads, generated directly (not through a shared/global index
// lookup) so that misusing a tag on the wrong `fm_` tuple fails to compile. Assumes a `fm_`
// FlatTuple member (a convention, not enforced by the macro).
#define EXADEM_FIELD_ACCESSOR_ITEM(TYPE, NAME, INDEX, MODE)                                       \
  ONIKA_HOST_DEVICE_FUNC inline TYPE& operator[](::exaDEM::attr::_##NAME) {                       \
    static_assert(INDEX < decltype(fm_)::size(), "field index out of range for fm_ (" #NAME ")"); \
    return fm_.get(onika::tuple_index<INDEX>);                                                    \
  }                                                                                               \
  ONIKA_HOST_DEVICE_FUNC inline const TYPE& operator[](::exaDEM::attr::_##NAME) const {           \
    static_assert(INDEX < decltype(fm_)::size(), "field index out of range for fm_ (" #NAME ")"); \
    return fm_.get(onika::tuple_index<INDEX>);                                                    \
  }

// Declares every operator[](attr::_NAME) overload from a field list LIST(X). Call once inside
// the struct body, regardless of how many fields are declared. MODE is not consulted here: each
// call site gets its own overloads (no risk of redeclaration between different structs).
#define EXADEM_INTERACTION_FIELD_ACCESSOR(LIST) LIST(EXADEM_FIELD_ACCESSOR_ITEM)

// Same as above, but for a `fm_` member holding VectorT<TYPE> per field instead of TYPE
// directly (see EXADEM_INTERACTION_VECTOR_TUPLE_TYPE).
#define EXADEM_FIELD_VECTOR_ACCESSOR_ITEM(TYPE, NAME, INDEX, MODE)                                \
  ONIKA_HOST_DEVICE_FUNC inline VectorT<TYPE>& operator[](::exaDEM::attr::_##NAME) {              \
    static_assert(INDEX < decltype(fm_)::size(), "field index out of range for fm_ (" #NAME ")"); \
    return fm_.get(onika::tuple_index<INDEX>);                                                    \
  }                                                                                               \
  ONIKA_HOST_DEVICE_FUNC inline const VectorT<TYPE>& operator[](::exaDEM::attr::_##NAME) const {  \
    static_assert(INDEX < decltype(fm_)::size(), "field index out of range for fm_ (" #NAME ")"); \
    return fm_.get(onika::tuple_index<INDEX>);                                                    \
  }
#define EXADEM_INTERACTION_VECTOR_FIELD_ACCESSOR(LIST) LIST(EXADEM_FIELD_VECTOR_ACCESSOR_ITEM)

// Same as EXADEM_INTERACTION_(VECTOR_)FIELD_ACCESSOR, but targets a second tuple member named
// `bk_` instead of `fm_`.
#define EXADEM_FIELD_ACCESSOR_ITEM_BK(TYPE, NAME, INDEX, MODE)                                    \
  ONIKA_HOST_DEVICE_FUNC inline TYPE& operator[](::exaDEM::attr::_##NAME) {                       \
    static_assert(INDEX < decltype(bk_)::size(), "field index out of range for bk_ (" #NAME ")"); \
    return bk_.get(onika::tuple_index<INDEX>);                                                    \
  }                                                                                               \
  ONIKA_HOST_DEVICE_FUNC inline const TYPE& operator[](::exaDEM::attr::_##NAME) const {           \
    static_assert(INDEX < decltype(bk_)::size(), "field index out of range for bk_ (" #NAME ")"); \
    return bk_.get(onika::tuple_index<INDEX>);                                                    \
  }
#define EXADEM_INTERACTION_FIELD_ACCESSOR_BK(LIST) LIST(EXADEM_FIELD_ACCESSOR_ITEM_BK)

#define EXADEM_FIELD_VECTOR_ACCESSOR_ITEM_BK(TYPE, NAME, INDEX, MODE)                             \
  ONIKA_HOST_DEVICE_FUNC inline VectorT<TYPE>& operator[](::exaDEM::attr::_##NAME) {              \
    static_assert(INDEX < decltype(bk_)::size(), "field index out of range for bk_ (" #NAME ")"); \
    return bk_.get(onika::tuple_index<INDEX>);                                                    \
  }                                                                                               \
  ONIKA_HOST_DEVICE_FUNC inline const VectorT<TYPE>& operator[](::exaDEM::attr::_##NAME) const {  \
    static_assert(INDEX < decltype(bk_)::size(), "field index out of range for bk_ (" #NAME ")"); \
    return bk_.get(onika::tuple_index<INDEX>);                                                    \
  }
#define EXADEM_INTERACTION_VECTOR_FIELD_ACCESSOR_BK(LIST) LIST(EXADEM_FIELD_VECTOR_ACCESSOR_ITEM_BK)
