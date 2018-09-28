//------------------------------------------------------------------------------
/// \file template_literal_operator.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Demonstrate template literal operator.
/// \ref Ch. 19 Special Operators, 19.2.6 User-defined Literals
///   The C++ Programming Language, 4th Ed., Stroustrup;
/// \details template literal operator is a literal operator that takes its
/// argument as a template parameter pack, rather than as a function argument.
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
///  g++ -std=c++14 virtual_dtors_main.cpp -o virtual_dtors_main
//------------------------------------------------------------------------------
#ifndef _TEMPLATE_LITERAL_OPERATOR_H_
#define _TEMPLATE_LITERAL_OPERATOR_H_

// Use namespaces to prevent clashes.
namespace Utilities
{

// Standard library reserves all suffixes not starting with an initial
// underscore, so define your suffixes starting with an underscore or risk your
// code breaking in the future.
//template <char...>
//constexpr int operator"" _b3(); // base 3, i.e., ternary

// helper functions.
constexpr int ipow(int x, int n) // x to the nth power for n >= 0
{
  return (n > 0) ? x * ipow(x, n - 1) : 1;
}

template <char c, char... tail> // peel off one ternary digit
constexpr int b3_helper()
{
  static_assert(c < '3', "not a ternary digit");
  return ipow(3, sizeof...(tail)) * (c - '0') + b3_helper<tail...>();
}

template <char c> // handle the single ternary digit case
constexpr int b3_helper()
{
  static_assert(c < '3', "not a ternary digit");
  return c;
}

template <char... chars>
constexpr int operator""_b3() // base 3, i.e. ternary
{
  return b3_helper<chars...>();
}

} // Utilities

#endif // _TEMPLATE_LITERAL_OPERATOR_H_
