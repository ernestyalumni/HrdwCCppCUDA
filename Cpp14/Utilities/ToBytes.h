//------------------------------------------------------------------------------
/// \file ToBytes.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Obtain a value of type To by reinterpreting the object
/// representation of from for underlying bytes.
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

#ifndef _UTILITIES_TO_BYTES_H_
#define _UTILITIES_TO_BYTES_H_

#include <cstring> // strerror
#include <iostream>
#include <stdexcept> // std::runtime_error
#include <string>
#include <system_error> 

namespace Utilities
{

} // namespace Utilities