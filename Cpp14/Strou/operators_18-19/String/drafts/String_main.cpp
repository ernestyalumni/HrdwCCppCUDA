//------------------------------------------------------------------------------
/// \file String_main.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Main driver function for String class.
/// \ref Sec. 19.3 A String Class.
///   The C++ Programming Language, 4th Ed., Stroustrup; Ch. 19 Special Ops
/// \details String provides value semantics, checked and unchecked access to
/// characters, stream I/O, support for range-for loops, equality operations,
/// and concatenation operators.
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
///  g++ -std=c++14 String_main.cpp -o String_main
//------------------------------------------------------------------------------
#include "String.h"

#include <cstring> // std::strcpy 
#include <iostream>
#include <stdexcept> // std::out_of_range 
#include <type_traits> // std::is_copy_constructible
#include <utility> // std::move // std::move means 

using Utilities::String;

int main()
{
  // StringDefaultConstructs
  {
    std::cout << "\n StringDefaultConstructs\n";
    std::cout << " is_default_constructible : " <<
      std::is_default_constructible<String>::value << '\n'; // 1
    std::cout << " is_trivially_default_constructible : " <<
      std::is_trivially_default_constructible<String>::value << '\n'; // 0
    std::cout << " is_nothrow_default_constructible : " <<
      std::is_nothrow_default_constructible<String>::value << '\n'; // 0

    String s; // default constructed
    String s1 {}; // default constructed
  }

  // StringConstructsFromCStyleArray
  {
    String s {"Euler"};
  }

  // StringCopyConstructs
  {
    std::cout << "\n StringCopyConstructs\n";
    std::cout << " is_copy_constructible : " <<
      std::is_copy_constructible<String>::value << '\n'; // 1
    std::cout << " is_trivially_copy_constructible : " <<
      std::is_trivially_copy_constructible<String>::value << '\n'; // 0
    std::cout << " is_nothrow_copy_constructible : " <<
      std::is_nothrow_copy_constructible<String>::value << '\n'; // 0

    String s {"Euler"};
    String s_copy {s}; // copy constructed
    String s_copy1 = s; // copy constructed

    // Use const qualifier
    std::cout << "\n  Use const qualifier \n";
    const String s_const {"reconnaissant"};
    const String s_const1 {s_const}; // copy constructed
    String s_2 {s_const}; // copy constructed
    String s_3 = s_const; // copy constructed 
    const String s_const2 = s_3; // copy constructed
  }

  // StringCopyAssigns
  {
    std::cout << "\n StringCopyAssigns\n";
    std::cout << " is_copy_assignable : " <<
      std::is_copy_assignable<String>::value << '\n'; // 1
    std::cout << " is_trivially_copy_assignable : " <<
      std::is_trivially_copy_assignable<String>::value << '\n'; // 0
    std::cout << " is_nothrow_copy_assignable : " <<
      std::is_nothrow_copy_assignable<String>::value << '\n'; // 0

    String s {"Euler"};
    String s_copy;
    s_copy = s; // copy assignable
  }

  // StringMoveConstructs
  {
    std::cout << "\n StringMoveConstructs\n";
    std::cout << " is_move_constructible : " <<
      std::is_move_assignable<String>::value << '\n'; // 1
    std::cout << " is_trivially_move_constructible : " <<
      std::is_trivially_move_constructible<String>::value << '\n'; // 0
    std::cout << " is_nothrow_move_constructible : " <<
      std::is_nothrow_move_constructible<String>::value << '\n'; // 0

    String s {"Euler"};
    // std::move means "give me an rvalue reference to argument"
    String s_moved {std::move(s)}; // move constructed
    String s_moved1 = std::move(s_moved); // move constructed

    const String s_const {std::move(s_moved1)}; // move constructed
  }

  // StringMoveAssigns
  {
    std::cout << "\n StringMoveAssigns\n";
    std::cout << " is_move_assignable : " <<
      std::is_move_assignable<String>::value << '\n'; // 1
    std::cout << " is_trivially_move_assignable : " <<
      std::is_trivially_move_assignable<String>::value << '\n'; // 0
    std::cout << " is_nothrow_move_assignable : " <<
      std::is_nothrow_move_assignable<String>::value << '\n'; // 0

    String s {"Euler"};
    // std::move means "give me an rvalue reference to argument"
    String s_moved; 
    s_moved = std::move(s); // move assign    
  }

  // StringDestructible
  {
    std::cout << "\n StringDestructible\n";
    std::cout << " is_destructible : " <<
      std::is_destructible<String>::value << '\n'; // 1
    std::cout << " is_trivially_destructible : " <<
      std::is_trivially_destructible<String>::value << '\n'; // 0
    std::cout << " is_nothrow_destructible : " <<
      std::is_nothrow_destructible<String>::value << '\n'; // 1
  }

  // StringAccessorsAccessElements
  {
    std::cout << "\n StringAccessorsAccessElements\n";
    const String s {"Descartes1234567890123456"};

    for (int i {0}; i < s.size(); ++i)
    {
      std::cout << i << ' ' << s[i] << ' ';
    }

    for (int i {0}; i < s.size(); ++i)
    {
      std::cout << s.at(i) << ' ';
    }
    
    try
    {
      std::cout << s.at(s.size()) << ' ';
    }
    catch (const std::out_of_range&)
    {
      std::cout << "\n Out of range access with at" << '\n';
    }

    const char* c_string_array {s.c_str()};
    for (int i {0}; i < s.size(); ++i)
    {
      std::cout << c_string_array[i] << ' ';
    }

    std::cout << " size() : " << s.size() << '\n';
    std::cout << " capacity() : " << s.capacity() << '\n';
  }

  // StringAdditionIncrementAddsCharactersAtTheEndOfString
  {
    String s {"Descartes1234567890123456"};
    s += '7';

    std::cout << " size() : " << s.size() << '\n';
    std::cout << " capacity() : " << s.capacity() << '\n';

    for (int i {0}; i < s.size(); ++i)
    {
      std::cout << s.at(i) << ' ';
    }
  }

  // StringFriendHelperFunctionForStreamIOPrintsString
  {
    std::cout << "\n StringFriendHelperFunctionForStreamIOPrintsString\n";

    String s {"Descartes1234567890123456"};
    std::cout << s << '\n';

    std::cin >> s;
    std::cout << s << '\n';

    std::cout << " size() : " << s.size() << '\n';
    std::cout << " capacity() : " << s.capacity() << '\n';    
  }

  // StringComparesWithEqualsAndNotEqualsAsExactSameString
  {
    std::cout << "\n StringComparesWithEqualsAndNotEqualsAsExactSameString\n";
    const String s {"Descartes1234567890123456"};
    const String s1 {"Descartes1234567890123456"};
    const String s2 {"Euler"};
    std::cout << " s == s1 : " << (s == s1) << '\n';
    std::cout << " s == s2 : " << (s == s2) << '\n';
    std::cout << " s1 == s2 : " << (s1 == s2) << '\n';

    std::cout << " s != s1 : " << (s != s1) << '\n';
    std::cout << " s != s2 : " << (s != s2) << '\n';
    std::cout << " s1 != s2 : " << (s1 != s2) << '\n';
  }

  // \ref 19.3.6 Using Our String
  {
    String s {"abcdefghij"};
    std::cout << s << '\n';
    s += 'k';
    s += 'l';
    s += 'm';
    s += 'n';
    std::cout << s << '\n';

    String s2 {"Hell"};
//    s2 += " and high water";
//    std::cout << s2 << '\n';

//    String s3 = "qwerty";
  //  s3 = s3;

  }
}
