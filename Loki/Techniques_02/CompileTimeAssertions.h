//------------------------------------------------------------------------------
/// \file CompileTimeAssertions.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Compile-time assertions.
/// \ref 2.1 Compile-Time Assertions. Ch. 2 Techniques. pp. 19
///   Andrei Alexandrescu. Modern C++: Design Generic Programming and Design
///   Patterns Applied.
/// \details 
/// \copyright If you find this code useful, feel free to donate directly and
/// easily at this direct PayPal link: 
///
/// https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=ernestsaveschristmas%2bpaypal%40gmail%2ecom&lc=US&item_name=ernestyalumni&currency_code=USD&bn=PP%2dDonationsBF%3abtn_donateCC_LG%2egif%3aNonHosted 
/// 
/// which won't go through a 3rd. party such as indiegogo, kickstarter, patreon.
/// Otherwise, I receive emails and messages on how all my (free) material on
/// physics, math, and engineering have helped students with their studies, and
/// I know what it's like to not have money as a student, but love physics (or 
/// math, sciences, etc.), so I am committed to keeping all my material 
/// open-source and free, whether or not sufficiently crowdfunded, under the 
/// open-source MIT license: feel free to copy, edit, paste, make your own
/// versions, share, use as you wish.    
/// Peace out, never give up! -EY
//------------------------------------------------------------------------------
/// COMPILATION TIPS:
///  g++ -std=c++14 Socket_main.cpp -o Socket_main
//------------------------------------------------------------------------------
#ifndef _COMPILE_TIME_ASSERTIONS_H_
#define _COMPILE_TIME_ASSERTIONS_H_

namespace Techniques
{

namespace CompileTime // Compile-Time Assertions
{

template <class To, class From>
To safe_reinterpret_cast(From& from)
{
  static_assert(sizeof(From) <= sizeof(To));

  return reinterpret_cast<To>(from);
}

template <bool> struct CompileTimeError;

template<> 
struct CompileTimeChecker<true> {};

template <bool> struct CompileTimeChecker
{
  CompileTimeChecker(...);
};

template<>
struct CompileTimeChecker<false> {};

} // namespace CompileTime

} // namespace Techniques

#endif // _COMPILE_TIME_ASSERTIONS_H_
