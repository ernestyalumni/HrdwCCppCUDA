//------------------------------------------------------------------------------
/// \file NarrowCast.h
/// \author Ernest Yeung
/// \brief .
/// \ref Stroustrup. C++ Programming Language. 4th Ed. pp. 299. Sec. 11.5.
///   Explicit Type Conversion
///-----------------------------------------------------------------------------
#ifndef _UTILITIES_NARROW_CAST_H_
#define _UTILITIES_NARROW_CAST_H_

#include <stdexcept> // std::runtime_error

namespace Utilities
{

template <class Target, class Source>
Target narrow_cast(Source v)
{
  auto r = static_cast<Target>(v); // convert value to target type
  if (static_cast<Source>(r) != v)
  {
    throw std::runtime_error("narrow_cast<>() failed");
  }
  return r;
}

} // namespace Utilities

#endif // _UTILITIES_NARROW_CAST_H_