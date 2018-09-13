//------------------------------------------------------------------------------
/// \file stdremove_reference.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Main driver file for examples of std::remove_reference
/// reference deduction. 
/// \ref 23.5.2.1 Reference Deduction Ch. 23 Templates; Bjarne Stroustrup, 
///   The C++ Programming Language, 4th Ed., Stroustrup; Ch.23
/// https://en.cppreference.com/w/cpp/types/remove_reference
/// \details If type T is a reference type, provides member typedef typedef
/// type which is the type referred to by T. Otherwise type is T.
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
#include <iostream> // std::cout 
#include <type_traits> // std::is_same, std::remove_reference

template <class T1, class T2>
void print_is_same()
{
  std::cout << std::is_same<T1, T2>() << '\n';
}

int main()
{
  std::cout << std::boolalpha;

  print_is_same<int, int>();  // true
  print_is_same<int, int&>(); // false
  print_is_same<int, int &&>(); // false

  // C++11 style first
//  print_is_same<int, std::remove_reference<int>::type>();
  print_is_same<int, std::remove_reference_t<int>>(); // true
  print_is_same<int, std::remove_reference_t<int &>>(); // true
  print_is_same<int, std::remove_reference_t<int &&>>(); // true

}

