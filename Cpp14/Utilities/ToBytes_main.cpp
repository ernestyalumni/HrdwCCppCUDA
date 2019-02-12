//------------------------------------------------------------------------------
/// \file ToBytes.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Obtain a value of type To by reinterpreting the object
/// representation of from for underlying bytes.
/// \ref https://en.cppreference.com/w/cpp/numeric/bit_cast
/// \copyright If you find this code useful, feel free to donate directly
/// (username ernestyalumni or email address above), going directly to:
///
/// paypal.me/ernestyalumni
///
/// which won't go through a 3rd. party like indiegogo, kickstarter, patreon.
/// Otherwise, I receive emails and messages on how all my (free) material on
/// physics, math, and engineering have helped students with their studies, and
/// I know what it's like to not have money as a student, but love physics (or
/// math, sciences, etc.), so I am committed to keeping all my material
/// open-source and free, whether or not sufficiently crowdfunded, under the
/// open-source MIT license: feel free to copy, edit, paste, make your own
/// versions, share, use as you wish.
/// Peace out, never give up! -EY
///
/// \details Every bit in the value representation of the returned To object is
/// equal to the corresponding bit in the object representation of from. The
/// values of padding bits in returned To object are unspecified.
//------------------------------------------------------------------------------
/// COMPILATION TIPS:
///  g++ -std=c++17 ToBytes_main.cpp -o ToBytes_main
//------------------------------------------------------------------------------
#include <array>
#include <cstdint> // int8_t, int16_t, ... uint8_t, uint16_t
#include <iostream>
#include <cstdio>
#include <vector>

int main()
{

  /// \url https://en.cppreference.com/w/cpp/types/integer
  /// int8_t, int16_t signed integer type with width exactly 8, 16 bits, ...
  /// resp.
  /// uint8_t, uint16_t unsigned integer type with width of exactly 8, 16, ...
  /// bits, resp.
  {
    std::cout << sizeof(int8_t) << ' ' << sizeof(int16_t) << ' ' << 
      sizeof(uint8_t) << ' ' << sizeof(uint16_t) << '\n';

    int8_t x {0x12};

    std::cout << "x : " << x << '\n';

    std::array<unsigned char, sizeof(int8_t)> a {
      reinterpret_cast<unsigned char&>(x)};

    for (auto i : a)
    {
      printf("%X", i);
    }

    std::cout << "\n";

    uint16_t x2 {0x7654};

    std::array<unsigned char, sizeof(uint16_t)> a2 {
      reinterpret_cast<unsigned char&>(x2)};
    

    for (auto i : a2)
    {
      printf("%X ", i);
    }

    std::vector<unsigned char> v;
//    v.assign(&(reinterpret_cast<unsigned char*>(&x2)[0]), sizeof(uint16_t));

  }
}




