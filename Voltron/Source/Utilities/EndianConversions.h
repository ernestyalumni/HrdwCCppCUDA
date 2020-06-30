//------------------------------------------------------------------------------
/// \file EndianConversions.h
/// \author Ernest Yeung
/// \brief .
/// \ref 
///-----------------------------------------------------------------------------
#ifndef UTILITIES_ENDIAN_CONVERSIONS_H
#define UTILITIES_ENDIAN_CONVERSIONS_H

#include "Utilities/ToHexString.h"

#include <boost/endian/conversion.hpp>

namespace Utilities
{

template <typename T>
ToHexString<T> to_big_endian(const ToHexString<T>& hex_string)
{
  ToHexString converted {hex_string};
  converted.value_ = boost::endian::native_to_big(converted.value_);
  return converted;
}

template <typename T>
ToHexString<T> to_little_endian(const ToHexString<T>& hex_string)
{
  ToHexString converted {hex_string};
  converted.value_ = boost::endian::native_to_little(converted.value_);
  return converted;
}

} // namespace Utilities

#endif // UTILITIES_ENDIAN_CONVERSIONS_H