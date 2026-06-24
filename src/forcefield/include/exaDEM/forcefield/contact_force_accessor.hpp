#pragma once

namespace exaDEM {
/** Common API to access contact force parameters. */

template <typename ContactParamsT>
struct SingleMatContactParamsTAccessor {
  const ContactParamsT cpt_;
  ONIKA_HOST_DEVICE_FUNC inline const ContactParamsT& operator()(int typeA, int typeB) const {
    return cpt_;
  }
};

template <typename ContactParamsT>
struct MultiMatContactParamsTAccessor {
  const ContactParamsT* cpt_;
  const int size_;
  ONIKA_HOST_DEVICE_FUNC inline const ContactParamsT& operator()(int typeA, int typeB) const {
    assert(typeA < size_);
    assert(typeB < size_);
    return cpt_[typeA * size_ + typeB];
  }
};
}  // namespace exaDEM
