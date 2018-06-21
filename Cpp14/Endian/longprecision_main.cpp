//------------------------------------------------------------------------------
/// \file longprecision_main.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Driver main file for Endian.h to test utility functions and classes
///   that deal with bytesex, i.e.Endianness.
/// \ref https://benjaminjurke.com/content/articles/2015/loss-of-significance-in-floating-point-computations/
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
#include <iostream>
#include <string>
#include <stdio.h>

//------------------------------------------------------------------------------
/// \details We use a generic function using variable input lengths to turn 
///   floating-point numbers into sequences of bits. Note that we need to 
///   reverse the order of bytes due to the Little-Endian arrangement of the 
///   data in memory on x86 and x64 architectures...
//------------------------------------------------------------------------------
std::string GetBitSequence(unsigned char* bytes, int len)
{
  std::string bits;
  bytes += len;   // advance 'len' bytes
  while (len--)
  {
    bytes--;  // go through the bytes in reverse order
    unsigned char bitmask {1 << 7}; // single byte bitmask
    for (int i {0}; i < 8; i++)
    {
      bits.append(*bytes & bitmask ? "1" : "0");
      bitmask >>= 1;
    }
  }
  return bits;
}


std::string GetDoubleBitsNice(double val)
{
  std::string bits {GetBitSequence((unsigned char*)&val, 8)};
  char buf[128];
  snprintf(buf, sizeof(buf), "%3.4f  =  %s %s %s", val,
    bits.substr(0, 1).c_str(),     // sign bit
    bits.substr(1, 11).c_str(),    // 11-bit exponent
    bits.substr(12, 52).c_str());  // 52-bit mantissa
  return std::string(buf);
}

static_assert(sizeof(long double) >= 10,
  "long double is not of 80-bit extended precision type");

std::string GetLongDoubleBitsNice(long double val)
{
  std::string bits {GetBitSequence((unsigned char*)&val, 10)};
  char buf[192];
  snprintf(buf, sizeof(buf), "%3.4f d  = %s %s %s %s b", (double) val,
    bits.substr(0, 1).c_str(),      // sign bit
    bits.substr(1, 15).c_str(),     // 15-bit exponent
    bits.substr(16, 1).c_str(),     // integer bit
    bits.substr(17, 63).c_str());   // 63-bit mantissa
  return std::string(buf);
}

int main()
{
  // LowLevelImplementationChecks
  unsigned char bitmask {1 << 7}; // 128

  std::cout << " bitmask: " << bitmask << " (1<<7) : " << (1 << 7) << '\n';


  double Dlarge {1.0};
  double Dsmall {1.0 / (double)(1ull << 53)};

  long double LDlarge {1.0};
  long double LDsmall {1.0 / (long double)(1ull << 53)};

  std::cout << "sizeof(double) = " << sizeof(double) << '\n';
  std::cout << "sizeof(long double) = " << sizeof(long double) << '\n';

  std::cout << GetDoubleBitsNice(Dlarge) << '\n';
  std::cout << GetDoubleBitsNice(Dsmall) << '\n'; // 2^(-53)
  std::cout << GetDoubleBitsNice(Dlarge + Dsmall + Dsmall + Dsmall + Dsmall
                                        + Dsmall + Dsmall + Dsmall + Dsmall
                                        + Dsmall + Dsmall + Dsmall + Dsmall
                                        + Dsmall + Dsmall + Dsmall + Dsmall) << '\n';
  std::cout << GetLongDoubleBitsNice(LDlarge) << '\n';
  std::cout << GetLongDoubleBitsNice(LDsmall) << '\n';
  std::cout << GetLongDoubleBitsNice(LDlarge + LDsmall) << std::endl;

  return 0;
}
