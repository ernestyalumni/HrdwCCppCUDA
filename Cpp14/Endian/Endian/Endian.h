//------------------------------------------------------------------------------
/// \file Endian.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Utility functions and classes that depend on bytesex, i.e.Endianness.
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
///  g++ -std=c++14 stdarray.cpp -o stdarray
//------------------------------------------------------------------------------
#ifndef ENDIAN_ENDIAN_H
#define ENDIAN_ENDIAN_H

#include <cassert>
#include <string>

namespace Endian
{

//------------------------------------------------------------------------------
/// \ref https://gist.github.com/khajavi/5667960
/// \details https://stackoverflow.com/questions/11197931/what-is-meaning-of-in-struct-c
/// ":" in struct is a bit field; basically tells compiler, "hey, this variable
/// only needs to be x bits wide, so pack the rest of the fields in accordingly"
//------------------------------------------------------------------------------
union FloatingPointIEEE754
{
  struct
  {
    unsigned int mantissa : 23;
    unsigned int exponent : 8;
    unsigned int sign : 1;
  } raw;
  float f;
} FloatRepresentationUnion;

union DoubleRepresentationUnion
{
  struct
  {
    unsigned long mantissa : 52;
    unsigned int exponent : 11;
    unsigned int sign : 1;
  } raw;
  double x;
};

//------------------------------------------------------------------------------
/// \details We use a generic function using variable input lengths to turn 
///   floating-point numbers into sequences of bits. Note that we need to 
///   reverse the order of bytes due to the Little-Endian arrangement of the 
///   data in memory on x86 and x64 architectures...
///
/// \ref https://benjaminjurke.com/content/articles/2015/loss-of-significance-in-floating-point-computations/
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

std::string GetDoubleBitsHex(double val)
{
  std::string bits {GetBitSequence((unsigned char*)&val, 8)};
  char buf[128];
  snprintf(buf, sizeof(buf), "%3.4f  =  %x %x %x", val,
    bits.substr(0, 1).c_str(),     // sign bit
    bits.substr(1, 11).c_str(),    // 11-bit exponent
    bits.substr(12, 52).c_str());  // 52-bit mantissa
  return std::string(buf);
}

//------------------------------------------------------------------------------
/// \brief Define ValueType->IntType mapping for the unified 
///   "IntType FromHost (ValueType)" API. The mapping is implemented via"

//------------------------------------------------------------------------------
/// \brief Utilities to convert numbers between current host's native byte
///   order and big-endian byte order (same as network byte order)
/// \details Load/Store methods are alignment safe
//------------------------------------------------------------------------------
class BigEndian
{
  public:

    //--------------------------------------------------------------------------
    /// \name Unified BigEndian::Load/Store<T> API
    /// \brief Returns the T value encoded by the leading bytes of 'p',
    ///   interpreted according to the format specified below. 'p' has no 
    ///   alignment restrictions.
    ///
    /// Type            Format
    /// ------------    --------------------------------------------------------
    /// float, double   Big-endian IEEE-754 format.
    //--------------------------------------------------------------------------
    template<typename T>
    static T Load(const char* p);

    //--------------------------------------------------------------------------
    /// \brief Encodes 'value' in the format corresponding to T. Supported 
    ///   types are described in Load<T>(). 'p' has no alignment restrictions. 
    ///   In-place Store is safe (that is, it is safe to call
    ///     Store(x, reinterpret_cast<char*>(&x)).
    //--------------------------------------------------------------------------
    template <typename T>
    static void Store(T value, char* p);

}; // BigEndian

} // namespace Endian

#endif // ENDIAN_ENDIAN_H
