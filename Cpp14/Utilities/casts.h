//------------------------------------------------------------------------------
/// \file casts.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Helper functions to cast objects.
/// \ref 
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
///  g++ -std=c++14 Clocks_main.cpp -o Clocks_main
//------------------------------------------------------------------------------
#ifndef _UTILITIES_CASTS_H_
#define _UTILITIES_CASTS_H_

#include <type_traits> // std::underlying_type

namespace Utilities
{

template <typename T>
constexpr std::underlying_type_t<T> get_underlying_value(const T& t)
{
  return static_cast<std::underlying_type_t<T>>(t);
}

} // namespace Utilities

#endif // _UTILITIES_CASTS_H_
