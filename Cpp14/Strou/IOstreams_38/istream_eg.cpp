//------------------------------------------------------------------------------
/// \file istream_eg.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  istream examples
/// \ref https://gist.github.com/ochinchina/e6d65abc2508e528084a     
/// \details istream examples. 
/// \copyright If you find this code useful, feel free to donate directly and easily at 
/// this direct PayPal link: 
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
///  g++ -std=c++14 istream_eg.cpp -o istream_eg
//------------------------------------------------------------------------------
#include <cassert> // assert
#include <cstddef> // std::size_t 
#include <iostream>
#include <istream> // std::istream
#include <sstream> // std::istringstream

//------------------------------------------------------------------------------
std::size_t readBytes(std::istream& in, char* buf, std::size_t len)
{
  std::size_t n {0};

  while (len > 0 && in.good())
  {
    in.read( &buf[n], len);
    long int i {in.gcount()};
    n += i;
    len -= i;
  }

  return n;
}

int main(int argc, char** argv)
{
  std::istringstream in {"this is a test stream"};
  char buf[100];

  std::size_t n {readBytes(in, buf, sizeof(buf))};
  assert(n < 100);

  std::cout << " n : " << n << std::endl;
}
