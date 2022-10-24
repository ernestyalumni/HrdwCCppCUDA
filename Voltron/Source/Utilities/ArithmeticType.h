#ifndef UTILITIES_ARITHMETIC_TYPE_H
#define UTILITIES_ARITHMETIC_TYPE_H

namespace Utilities
{
namespace Kedyk
{

//------------------------------------------------------------------------------
/// \ref pp. 40. Ch. 5 Fundamental Data Structures.
/// \details Another useful functionality is some common arithmetic operators.
/// In many cases it's standard to implement operator+ in terms of operator+=.
/// With templates friend is often less clumsy than a nonmember.
//------------------------------------------------------------------------------

template <typename T>
struct ArithmeticType
{
  friend T operator+(const T& a, const T& b)
  {
    T result {a};
    return result += b;
  }

  friend T operator-(const T& a, const T& b)
  {
    T result {a};
    return result -= b;
  }

  friend T operator*(const T& a, const T& b)
  {
    T result {a};
    return result *= b;
  }

  friend T operator<<(const T& a, const int shift)
  {
    T result {a};
    return result <<= shift;
  }

  friend T operator>>(const T& a, const int shift)
  {
    T result {a};
    return result >>= shift;
  }

  friend T operator%(const T& a, const T& b)
  {
    T result {a};
    return result %= b;
  }

  friend T operator/(const T& a, const T& b)
  {
    T result {a};
    return result /= b;
  }
};

} // namespace Kedyk
} // namespace Utilities

#endif // UTILITIES_ARITHMETIC_TYPE_H