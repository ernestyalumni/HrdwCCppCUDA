//------------------------------------------------------------------------------
/// \file Enums.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Demonstrate Enums in a header. 
/// \ref 8.4 Enumerations Ch. 8 STructures, Unions, and Enumerations;
///   Bjarne Stroustrup, The C++ Programming Language, 4th Ed.       
/// \details An enumeration is a type that can hold a set of integer values
///   specified by the user.
/// A prvalue of an unscoped enumeration type whose underlying type is fixed 
/// (10.2) can be converted to a prvalue of its underlying type. - 7.6 
/// Integral promotions, 7.6.4, ISO n4713.pdf
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
///  g++ -std=c++14 Enums_main.cpp -o Enums_main
//------------------------------------------------------------------------------
#ifndef _ENUMS_H_
#define _ENUMS_H_

#include <iostream>
#include <string>
#include <type_traits> // std::remove_reference

namespace Types
{

namespace Enums
{

// An enumeration is a distinct type (6.7.2) with named constants. - 10.2.1
// Enumeration declarations [dcl.enum], ISO, n4713.pdf

enum class S
{
  s_a,
  s_b,
  s_c
};

enum class S_ZZ : int
{
  s_a,
  s_b = 1,
  s_c
};

// warning : elaborated-type-specifier for a scoped enum must not use the
// ‘class’ keyword
/*
enum class S_ZZ32_2 : int32_t
{
  s_a,
  s_b = 2,
  s_c
};*/

// error: use of enum ‘S_ZZ32_2’ without previous declaration
/*enum S_ZZ32_2 : int32_t
{
  s_a,
  s_b = 2,
  s_c
};*/


enum class S_NN : unsigned int
{
  s_a,
  s_b = 2,
  s_c
};

// warning: elaborated-type-specifier for a scoped enum must not use the
// ‘class’ keyword
/*enum class S_NN32_2 : uint32_t
{
  s_a,
  s_b = 1,
  s_c
};*/

enum class S_ZZ8_2 : char
{
  s_a,
  s_b = 0x01,
  s_c
};

enum class S_NN8_2 : unsigned char
{
  s_a,
  s_b = 0x02,
  s_c
};

// error: underlying type ‘float’ of ‘Types::Enums::S_RR32’ must be an
// integral type
/*enum class S_RR32 : float
{
  s_a,
  s_b,
  s_c
};*/ 

enum class T_ZZ8_2 : unsigned char
{
  s_a = 'a',
  s_b = 'b',
  s_c = 'c',
};

// It's possible to declare an enum class without defining it until later.
enum class Spin : bool;
/*
enum class Spin : bool
{
  up = true,
  down = false
};
*/

// Values for enumerator can be chosen so they can be combined by bitwise
// operations.
// pp .220, Ch. 8 Structures, Unions, and Enumerations.
enum class Printer_flags 
{
  acknowledge = 1,
  paper_empty = 2,
  busy = 4,
  out_of_black = 8,
  out_of_color = 16
};


enum class ByteFlags : char
{
  s_0 = 0x00,
  s_1 = 0x01,
  s_2 = 0x02,
  s_3 = 0x04,
  s_4 = 0x08,
  s_5 = 0x10
};

constexpr Printer_flags operator|(Printer_flags a, Printer_flags b)
{
  return static_cast<Printer_flags>(static_cast<int>(a) | static_cast<int>(b));
}

constexpr Printer_flags operator&(Printer_flags a, Printer_flags b)
{
  return static_cast<Printer_flags>(static_cast<int>(a) & static_cast<int>(b));
}

/*
void try_to_print(Printer_flags x)
{
  if (x & Printer_flags::acknowledge) // not a bool
  {
    std::cout << " Printer_flag acknowledge \n";
  }
  else if (x & Printer_flags::paper_empty)
  {
    std::cout << " Printer_flag paper_empty \n";
  }
  else if (x & Printer_flags::busy)
  {
    std::cout << " Printer_flag busy \n";
  }
  else if (x & Printer_flags::out_of_paper)
  {
    std::cout << " Printer_flag out_of_paper \n";
  }
  else if (x & Printer_flags::out_of_color)
  {
    std::cout << " Printer_flag out_of_color \n";
  }
}
*/

} // namespace Enums

} // namespace Types

#endif // _ENUMS_H_
