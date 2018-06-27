//------------------------------------------------------------------------------
/// \file String.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  More general type string that relies on fact that a character can
///   be copied.
/// \details The basic mechanisms for defining and using class templates are
///   introduced through the example of a string template.
/// \ref 23.2 A Simple String Template Ch. 23 Templates; Bjarne Stroustrup, 
///   The C++ Programming Language, 4th Ed., Stroustrup; Ch.23  
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
///  g++ -std=c++14 FileOpen_main.cpp FileOpen.cpp -o FileOpen_main
//------------------------------------------------------------------------------
#ifndef _STRING_H_
#define _STRING_H_

#include <array>
#include <cstddef> // std::size_t, for std::array
#include <cstring>
#include <string>

namespace Templates
{

namespace Strings
{

//------------------------------------------------------------------------------
/// \brief More general string type, making the character type a parameter. 
//------------------------------------------------------------------------------
// template <typename C> prefix specifies that a template is being declared and
// that a type argument C will be used in the declaration.
template <typename C>
class String
{
  public:

    String();

    explicit String(const C*);

    String(const String&);

    String operator=(const String&);

    // unchecked element access
    C& operator[](int n)
    {
      return ptr_[n];
    }

    // add c at end
    String& operator+=(C c);

  private:

    static const int short_max_ {15}; // for the short string optimization
    int sz_;
    C* ptr_; // ptr points to sz Cs
};

//------------------------------------------------------------------------------
/// \ref https://en.cppreference.com/w/cpp/string/basic_string
//------------------------------------------------------------------------------

template <
  typename C,
  class Traits = std::char_traits<C>,
  class Allocator = std::allocator<C>>
class BasicString : public std::basic_string<C, Traits, Allocator>
{
  public:
};

} // Strings

} // Templates

#endif // _STRING_H_
