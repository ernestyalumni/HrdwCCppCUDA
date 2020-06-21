//------------------------------------------------------------------------------
/// \file StdBitCast_tests.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \details Function template that obtains value of type by reinterpreting
/// object representation of From.
/// \ref https://en.cppreference.com/w/cpp/numeric/bit_cast
//------------------------------------------------------------------------------
#ifndef CPP_NUMERICS_BIT_CAST_H
#define CPP_NUMERICS_BIT_CAST_H

#include <cstring> // std::memcpy
#include <type_traits>

namespace Cpp
{
namespace Numerics
{

#ifdef __cpp_lib_bit_cast

#include <bit>

using std::bit_cast;

#else

// cf. https://en.cppreference.com/w/cpp/numeric/bit_cast
// Copied implementation of bit_cast from cppreference.com

// reinterpret_cast (or equivalent explicit cast) between pointer or reference
// types shall not be used to reinterpret object representation in most cases
// because of type aliasing rule. TODO-understand type aliasing.
template <class To, class From>
typename std::enable_if_t<
  (sizeof(To) == sizeof(From)) &&
  std::is_trivially_copyable<From>::value &&
  std::is_trivial<To>::value &&
  (std::is_copy_constructible<To>::value ||
    std::is_move_constructible<To>::value),
  // this implementation requires that To is trivially default constructible
  // and that To is copy constructible or move constructible.
  To>
// constexpr support needs compiler magic
bit_cast(const From &source) noexcept
{
  To destination;

  // void* memcpy(void* dest, const void* src, std::size_t count)
  // Copies count bytes from object pointed to by src to object pointed to by
  // dest. Both objects are reinterpred as arrays of unsigned char.
  // If objects overlap, behavior is undefined.
  // If either dest or src is an invalid or null pointer, behavior is undefined,
  // even if count is 0.
  std::memcpy(&destination, &source, sizeof(To));
  return destination;
}

#endif // #ifdef __cpp_lib_bit_cast

} // namespace Numerics
} // namespace Cpp

#endif // CPP_NUMERICS_BIT_CAST_H