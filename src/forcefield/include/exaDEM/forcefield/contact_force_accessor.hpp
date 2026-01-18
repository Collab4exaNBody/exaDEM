#pragma once

namespace exaDEM {
/** Common API to access contact force parameters. */

template <typename ContactParamsT>
struct SingleMatContactParamsTAccessor {
  const ContactParamsT cpt;
  ONIKA_HOST_DEVICE_FUNC inline const ContactParamsT& operator()(int typeA, int typeB) const {
    return cpt;
  }
};

template <typename ContactParamsT>
struct MultiMatContactParamsTAccessor {
  const ContactParamsT* cpt;
  const int size;
  ONIKA_HOST_DEVICE_FUNC inline const ContactParamsT& operator()(int typeA, int typeB) const {
    assert(typeA < size);
    assert(typeB < size);
    return cpt[typeA * size + typeB];
  }
};
}  // namespace exaDEM
