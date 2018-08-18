//------------------------------------------------------------------------------
/// \file Enums_main.cpp
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
#include "Enums.h"

#include <iostream> // std::boolalpha
#include <type_traits>

using Types::Enums::S;
using Types::Enums::S_ZZ;
//using Types::Enums::S_ZZ32_2;
using Types::Enums::S_NN;
//using Types::Enums::S_NN32_2;
using Types::Enums::S_ZZ8_2;
using Types::Enums::S_NN8_2;
using Types::Enums::Printer_flags;
using Types::Enums::ByteFlags;
using Types::Enums::operator|;
using Types::Enums::operator&;
//using Types::Enums::try_to_print;

// it's possible to declare an enum class without defining it (Sec. 6.3) until
// later.
enum class Spin : bool
{
  up = true,
  down = false
};

void print_spin(Spin spin)
{
  switch (spin)
  {
    case Spin::up:
      std::cout << " Spin up" << '\n';      
      break; // needed to break!
    case Spin::down:
      std::cout << " Spin down" << '\n';      
      break;  // needed to break!
  }
}

int main()
{
  // EnumClassesThemselvesDefaultConstruct;
  S s_test;

  // EnumClassesRuleOf5Results
  std::cout << std::boolalpha << 
    (std::is_default_constructible<S>::value == true) << '\n';
  std::cout << std::boolalpha << 
    (std::is_trivially_default_constructible<S>::value == true) << '\n';
  std::cout << std::boolalpha << 
    (std::is_nothrow_default_constructible<S>::value == true) << '\n';

  std::cout << std::boolalpha << 
    (std::is_copy_constructible<S>::value == true) << '\n';
  std::cout << std::boolalpha << 
    (std::is_trivially_copy_constructible<S>::value == true) << '\n';
  std::cout << std::boolalpha << 
    (std::is_nothrow_copy_constructible<S>::value == true) << '\n';

  std::cout << std::boolalpha << 
    (std::is_move_constructible<S>::value == true) << '\n';
  std::cout << std::boolalpha << 
    (std::is_trivially_move_constructible<S>::value == true) << '\n';
  std::cout << std::boolalpha << 
    (std::is_nothrow_move_constructible<S>::value == true) << '\n';

  std::cout << std::boolalpha << 
    (std::is_copy_assignable<S>::value == true) << '\n';
  std::cout << std::boolalpha << 
    (std::is_trivially_copy_assignable<S>::value == true) << '\n';
  std::cout << std::boolalpha << 
    (std::is_nothrow_copy_assignable<S>::value == true) << '\n';

  std::cout << std::boolalpha << 
    (std::is_move_assignable<S>::value == true) << '\n';
  std::cout << std::boolalpha << 
    (std::is_trivially_move_assignable<S>::value == true) << '\n';
  std::cout << std::boolalpha << 
    (std::is_nothrow_move_assignable<S>::value == true) << '\n';

  std::cout << std::boolalpha << 
    (std::is_destructible<S>::value == true) << '\n';
  std::cout << std::boolalpha << 
    (std::is_trivially_destructible<S>::value == true) << '\n';
  std::cout << std::boolalpha << 
    (std::is_nothrow_destructible<S>::value == true) << '\n';

  // EnumClassItselfKnowsItsType
  std::cout << std::boolalpha << 
    (std::is_integral<S>::value == false) << '\n';
  std::cout << std::boolalpha << 
    (std::is_enum<S>::value == true) << '\n';
  std::cout << std::boolalpha << 
    (std::is_class<S>::value == false) << '\n';
  std::cout << std::boolalpha << 
    (std::is_fundamental<S>::value == false) << '\n';
  std::cout << std::boolalpha << 
    (std::is_object<S>::value == true) << '\n';
  std::cout << std::boolalpha << 
    (std::is_trivial<S>::value == true) << '\n';
  std::cout << std::boolalpha << 
    (std::is_standard_layout<S>::value == true) << '\n';


  std::cout << std::boolalpha << 
    (std::is_default_constructible<S_ZZ>::value == true) << '\n';
  std::cout << std::boolalpha << 
    (std::is_trivially_default_constructible<S_ZZ>::value == true) << '\n';
  std::cout << std::boolalpha << 
    (std::is_nothrow_default_constructible<S_ZZ>::value == true) << '\n';

  std::cout << std::boolalpha << 
    (std::is_copy_constructible<S_ZZ>::value == true) << '\n';
  std::cout << std::boolalpha << 
    (std::is_trivially_copy_constructible<S_ZZ>::value == true) << '\n';
  std::cout << std::boolalpha << 
    (std::is_nothrow_copy_constructible<S_ZZ>::value == true) << '\n';

  std::cout << std::boolalpha << 
    (std::is_move_constructible<S_ZZ>::value == true) << '\n';
  std::cout << std::boolalpha << 
    (std::is_trivially_move_constructible<S_ZZ>::value == true) << '\n';
  std::cout << std::boolalpha << 
    (std::is_nothrow_move_constructible<S_ZZ>::value == true) << '\n';

  std::cout << std::boolalpha << 
    (std::is_copy_assignable<S_ZZ>::value == true) << '\n';
  std::cout << std::boolalpha << 
    (std::is_trivially_copy_assignable<S_ZZ>::value == true) << '\n';
  std::cout << std::boolalpha << 
    (std::is_nothrow_copy_assignable<S_ZZ>::value == true) << '\n';

  std::cout << std::boolalpha << 
    (std::is_move_assignable<S_ZZ>::value == true) << '\n';
  std::cout << std::boolalpha << 
    (std::is_trivially_move_assignable<S_ZZ>::value == true) << '\n';
  std::cout << std::boolalpha << 
    (std::is_nothrow_move_assignable<S_ZZ>::value == true) << '\n';

  std::cout << std::boolalpha << 
    (std::is_destructible<S_ZZ>::value == true) << '\n';
  std::cout << std::boolalpha << 
    (std::is_trivially_destructible<S_ZZ>::value == true) << '\n';
  std::cout << std::boolalpha << 
    (std::is_nothrow_destructible<S_ZZ>::value == true) << '\n';

  // EnumClassItselfKnowsItsType
  std::cout << std::boolalpha << 
    (std::is_integral<S_ZZ>::value == false) << '\n';
  std::cout << std::boolalpha << 
    (std::is_enum<S_ZZ>::value == true) << '\n';
  std::cout << std::boolalpha << 
    (std::is_class<S_ZZ>::value == false) << '\n';
  std::cout << std::boolalpha << 
    (std::is_fundamental<S_ZZ>::value == false) << '\n';
  std::cout << std::boolalpha << 
    (std::is_object<S_ZZ>::value == true) << '\n';
  std::cout << std::boolalpha << 
    (std::is_trivial<S_ZZ>::value == true) << '\n';
  std::cout << std::boolalpha << 
    (std::is_standard_layout<S_ZZ>::value == true) << '\n';

  std::cout << std::boolalpha <<
    (std::is_unsigned<S>::value == false) << '\n'; // true
  std::cout << std::boolalpha <<
    (std::is_unsigned<S_ZZ>::value == false) << '\n';
  std::cout << std::boolalpha <<
    (std::is_unsigned<S_NN>::value == false) << '\n';
  std::cout << std::boolalpha <<
    (std::is_unsigned<S_ZZ8_2>::value == false) << '\n';
  std::cout << std::boolalpha <<
    (std::is_unsigned<S_NN8_2>::value == false) << '\n';
  std::cout << std::boolalpha <<
    (std::is_unsigned<Spin>::value == false) << '\n';

  std::cout << std::boolalpha <<
    (std::is_signed<S>::value == false) << '\n'; // true
  std::cout << std::boolalpha <<
    (std::is_signed<S_ZZ>::value == false) << '\n';
  std::cout << std::boolalpha <<
    (std::is_signed<S_NN>::value == false) << '\n';
  std::cout << std::boolalpha <<
    (std::is_signed<S_ZZ8_2>::value == false) << '\n';
  std::cout << std::boolalpha <<
    (std::is_signed<S_NN8_2>::value == false) << '\n';
  std::cout << std::boolalpha <<
    (std::is_signed<Spin>::value == false) << '\n';


  std::cout << "\n EnumClassesSizesAreOfUnderlyingType : \n";
  // EnumClassSizesAReofUnderlyingType
  std::cout << std::boolalpha << 
    (sizeof(std::underlying_type_t<S>) == sizeof(int32_t)) << '\n';
  std::cout << std::boolalpha << 
    (sizeof(std::underlying_type_t<S_ZZ>) == sizeof(int32_t)) << '\n';
  std::cout << std::boolalpha << 
    (sizeof(std::underlying_type_t<S_NN>) == sizeof(int32_t)) << '\n';
  std::cout << std::boolalpha << 
    (sizeof(std::underlying_type_t<S_ZZ8_2>) == sizeof(int8_t)) << '\n';
  std::cout << std::boolalpha << 
    (sizeof(std::underlying_type_t<S_NN8_2>) == sizeof(int8_t)) << '\n';

  // this may not be the case; bool size is compiler-dependent.
  std::cout << std::boolalpha << 
    (sizeof(std::underlying_type_t<Spin>) == sizeof(char)) << '\n'; 


  std::cout << "\n EnumClassesCanConvertValue : \n";
  // EnumClassesCanConvertValue
  S_NN s_nn;
//  s_nn = S_NN{3}; // error: no narrowing conversion to an enum class
  s_nn = static_cast<S_NN>(3);
  std::cout << std::boolalpha << (s_nn ==  S_NN::s_c) << '\n';

  std::cout << "\n EnumClassesUsedForSwitchCases \n";
  // EnumClassesUsedForSwitchCases
  print_spin(Spin::up);
  print_spin(Spin::down);

  // EnumClassOverloadsBitwiseOperators
  Printer_flags printer_flag {
    Printer_flags::acknowledge & Printer_flags::acknowledge};

}
