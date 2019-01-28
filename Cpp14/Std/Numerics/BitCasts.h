//------------------------------------------------------------------------------
/// \file BitCasts.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Obtain a value of type To by reinterpreting the object
/// representation of from.
/// \ref https://en.cppreference.com/w/cpp/numeric/bit_cast
/// \copyright If you find this code useful, feel free to donate directly
/// (username ernestyalumni or email address above), going directly to:
///
/// paypal.me/ernestyalumni
///
/// which won't go through a 3rd. party like indiegogo, kickstarter, patreon.
/// Otherwise, I receive emails and messages on how all my (free) material on
/// physics, math, and engineering have helped students with their studies, and
/// I know what it's like to not have money as a student, but love physics (or
/// math, sciences, etc.), so I am committed to keeping all my material
/// open-source and free, whether or not sufficiently crowdfunded, under the
/// open-source MIT license: feel free to copy, edit, paste, make your own
/// versions, share, use as you wish.
/// Peace out, never give up! -EY
///
/// \details Every bit in the value representation of the returned To object is
/// equal to the corresponding bit in the object representation of from. The
/// values of padding bits in returned To object are unspecified.
//------------------------------------------------------------------------------
/// COMPILATION TIPS:
///  g++ -std=c++14 List_main.cpp -o List_main
//------------------------------------------------------------------------------
#ifndef _STD_NUMERICS_BIT_CASTS_H_
#define _STD_NUMERICS_BIT_CASTS_H_

#include <cstring> // std::cstring
#include <type_traits>

namespace Std
{
namespace Numerics
{

//------------------------------------------------------------------------------
/// \details This overload only participates in overload resolution if
/// sizeof(To) == sizeof(From) and both To and From are TriviallyCopyable types.
//------------------------------------------------------------------------------
template <class To, class From>
std::enable_if_t<
  (sizeof(To) == sizeof(From)) &&
  std::is_trivially_copyable<From>::value &&
  std::is_trivial<To>::value,
  // this implementation requires that To is trivially default constructible
  To>
// constexpr support needs compiler magic
bit_cast(const From &source) noexcept
{
  To destination;

  // \url https://en.cppreference.com/w/cpp/string/byte/memcpy
  // Both objects are reinterpreted as arrays of unsigned char
  std::memcpy(&destination, &source, sizeof(To));

  return destination;
}


} // namespace Numerics
} // namespace Std

#endif // _STD_NUMERICS_BIT_CASTS_H_
