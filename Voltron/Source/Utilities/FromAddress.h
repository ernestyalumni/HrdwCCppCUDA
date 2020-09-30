//------------------------------------------------------------------------------
/// \file FromAddress.h
/// \author Ernest Yeung
/// \brief Address conversion functions and classes.
/// \ref 
///-----------------------------------------------------------------------------
#ifndef UTILITIES_FROM_ADDRESS_H
#define UTILITIES_FROM_ADDRESS_H

#include "Utilities/ToBytes.h"
#include "Utilities/ToHexString.h"

#include <memory> // std::addressof
#include <sstream> // std::stringstream
#include <string>
#include <utility> // std::move

namespace Utilities
{

namespace FromAddress
{

template <typename PtrType, int N = 16>
ToBytes<unsigned long> from_address_of_to_bytes(PtrType ptr)
{
  std::stringstream string_stream;

  string_stream << std::addressof(ptr);

  unsigned long address_of_ptr {std::stoul(string_stream.str(), 0, N)};

  return ToBytes{address_of_ptr};
}

template <typename PtrType, int N = 16>
ToBytes<unsigned long> from_pointer_to_address_in_bytes(PtrType ptr)
{
  std::stringstream string_stream;

  string_stream << &ptr;

  unsigned long reference_of_ptr {std::stoul(string_stream.str(), 0, N)};

  return ToBytes{reference_of_ptr};
}

template <typename PtrType, int N = 16>
ToHexString<unsigned long> from_address_of_to_hex_string(PtrType ptr)
{
  std::stringstream string_stream;

  string_stream << std::addressof(ptr);

  unsigned long address_of_ptr {std::stoul(string_stream.str(), 0, N)};

  return ToHexString{std::move(address_of_ptr)};
}

template <typename PtrType, int N = 16>
ToHexString<unsigned long> from_pointer_to_address_in_hex_string(PtrType ptr)
{
  std::stringstream string_stream;

  string_stream << &ptr;

  unsigned long reference_of_ptr {std::stoul(string_stream.str(), 0, N)};

  return ToHexString{std::move(reference_of_ptr)};
}


} // namespace FromAddress

} // namespace Utilities

#endif // UTILITIES_FROM_ADDRESS_H