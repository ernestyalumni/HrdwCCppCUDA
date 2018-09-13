//------------------------------------------------------------------------------
/// \file Endian_main.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Driver main file for Endian.h to test utility functions and classes
///   that deal with bytesex, i.e.Endianness.
/// \ref https://github.com/google/sensei/blob/master/sensei/util/endian.h
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
///  g++ -std=c++14 Endian_main.cpp -o Endian_main
//------------------------------------------------------------------------------
#include "Endian/Endian.h"
#include <cmath> // std::nan

#include <iostream>

using namespace Endian;

// Trivially Copyable Examples
char a;
char b[5];

struct s1 
{
  char c[5];
  int d;
};

class c1:
  public s1
{
  protected:
    int a;
  public:
    c1():
      a{7}
    {}
};

union u1
{
  s1 s;
  c1 c;
};


// Not Trivially Copyable 
// User supplied copy ctor
struct s2
{
  s2 (s2 const& s):
    ch {'q'}
  {}

  char ch;
};

class c2
{
  virtual void f() // Virtual member
  {}
};


int main()
{
  const double pi {3.1415926535897932384626433};

  std::cout << " pi : " << pi << " pi (hex) : " << std::hex << pi << '\n';

  std::cout << GetDoubleBitsNice(pi) << '\n';

  std::cout << GetDoubleBitsHex(pi) << '\n';

  std::cout << std::hexfloat << pi << '\n';

  DoubleRepresentationUnion pi_union;
  pi_union.x = pi;
  std::cout << pi_union.raw.mantissa << ' ' << pi_union.raw.exponent << ' ' << 
    ' ' << pi_union.raw.sign << '\n';

  // NansPlayground
  // http://en.cppreference.com/w/cpp/numeric/math/nan
  double f1 {std::nan("1")};
  std::cout << GetDoubleBitsNice(f1) << '\n';

  std::cout << GetDoubleBitsHex(f1) << '\n';

  std::cout << std::hexfloat << f1 << '\n';

  DoubleRepresentationUnion nan_union;
  nan_union.x = f1;
  std::cout << nan_union.raw.mantissa << ' ' << nan_union.raw.exponent << ' ' << 
    ' ' << nan_union.raw.sign << '\n';

  double f2 {std::nan("2")};
  std::cout << GetDoubleBitsNice(f2) << '\n';

  std::cout << GetDoubleBitsHex(f2) << '\n';

  std::cout << std::hexfloat << f2 << '\n';

  nan_union.x = f2;
  std::cout << nan_union.raw.mantissa << ' ' << nan_union.raw.exponent << ' ' << 
    ' ' << nan_union.raw.sign << '\n';


  // \ref https://github.com/CppCon/CppCon2017/blob/master/Presentations/Type%20Punning%20In%20C%2B%2B17%20-%20Avoiding%20Pun-defined%20Behavior/Type%20Punning%20In%20C%2B%2B17%20-%20Avoiding%20Pun-defined%20Behavior%20-%20Scott%20Schurr%20-%20CppCon%202017.pdf
  // Scott Schurr, "Type Punning in C++17 - Avoiding Pun-defined Behavior"
//  static_assert(std::is_trivially_copyable<T>::value)  

  // Trivially Copyable Examples
  c1 a1[7];

  // Not Trivially Copyable
  char& r1(a); // Reference

  // Examples of Modifying const [dcl.type.cv] Sec. 4

  // initialized as required
  const int* ciq = new const int (3);

  const int* ciq2 {new const int {3}};

  std::cout << " ciq2 : " << ciq2 << " *ciq2 : " << *ciq2 << " ciq2[0] : " <<
    ciq2[0] << '\n';

  // cast required
  int* iq = const_cast<int*>(ciq2);

  std::cout << " iq : " << iq << " *iq : " << *iq << " iq[0] : " << iq[0] <<
    '\n';

  // undefined: modifies a const object
  *iq = 4;

  std::cout << " ciq2 : " << ciq2 << " *ciq2 : " << *ciq2 << " ciq2[0] : " <<
    ciq2[0] << '\n';

  std::cout << " iq : " << iq << " *iq : " << *iq << " iq[0] : " << iq[0] <<
    '\n';


}
