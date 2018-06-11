//------------------------------------------------------------------------------
/// \file stdarray.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  std::array examples
/// \ref https://en.cppreference.com/w/cpp/container/array     
/// \details std::array. 
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
///  g++ -std=c++14 stdarray.cpp -o stdarray
//------------------------------------------------------------------------------

#include <array>
#include <string>
#include <iostream>
#include <cstring> // std::strcpy

int main()
{
  // std::array as an array of chars.
  constexpr const std::size_t BUFLEN {2048};

  std::array<char, BUFLEN> buffer;
  std::array<char, 10> server {"127.0.0.1"};

  // StdArrayOfCharsCanPrintContentsByStdCout
  std::cout << " server.data() : " << server.data() << '\n';

  // StdArrayOfCharsCanBeSetByString
  std::string string_message {"This is a packet"};

// Doesn't work, buffer.data() is not l-value for left-operand 
//  buffer.data() = string_message.c_str();
  std::strcpy(buffer.data(), string_message.c_str());
  for (auto iter {buffer.begin()}; iter < buffer.begin() + 13; iter++)
  {
    std::cout << *iter << ' ';
  }

  // StdArrayOfCharsCanBeSetByStringLiteral
//  buffer.data() = "This is a packet 1";

}

