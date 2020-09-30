//------------------------------------------------------------------------------
/// \file ToBytes.h
/// \author Ernest Yeung
/// \brief .
/// \ref 
///-----------------------------------------------------------------------------
#ifndef UTILITIES_TO_BYTES_H
#define UTILITIES_TO_BYTES_H

#include <cstddef> // std::size_t
#include <cstdio> // printf
#include <string>

namespace Utilities
{

template <typename T>
class ToBytes
{
  public:

    ToBytes() = delete;

    ToBytes(T& x);

    ToBytes(T&& x);

    // TODO: deal with references vs. values
//    ToBytes(T x);

    void increasing_addresses_print() const;

    void decreasing_addresses_print() const;

    std::string increasing_addresses_hex() const;

    std::string decreasing_addresses_hex() const;

    void set(T& x);

    T data() const
    {
      return x_;
    }

  private:

    T x_;
};

template <typename T>
ToBytes<T>::ToBytes(T& x):
  x_{x}
{}

template <typename T>
ToBytes<T>::ToBytes(T&& x):
  x_{x}
{}


//template <typename T>
//ToBytes<T>::ToBytes(T x):
//  x_{x}
//{}

template <typename T>
void ToBytes<T>::increasing_addresses_print() const
{
  constexpr std::size_t number_of_bytes {sizeof(T)};

  auto x_as_chars = reinterpret_cast<const unsigned char*>(&x_);

  for (std::size_t i {0}; i < number_of_bytes; ++i)
  {
    // https://en.cppreference.com/w/cpp/io/c/fprintf
    // x X converts an unsigned integer into hexadecimal representation hhhh
    // For the x, conversion letters abcdef are used.
    // For the X, conversion letters ABCDEF are used.
    // 
    printf("%01x ", x_as_chars[i]);
  }
}

template <typename T>
std::string ToBytes<T>::increasing_addresses_hex() const
{
  constexpr std::size_t number_of_bytes {sizeof(T)};

  auto x_as_chars = reinterpret_cast<const unsigned char*>(&x_);

  std::string result;

  for (std::size_t i {0}; i < number_of_bytes; ++i)
  {
    // https://en.cppreference.com/w/cpp/io/c/fprintf
    // x X converts an unsigned integer into hexadecimal representation hhhh
    // For the x, conversion letters abcdef are used.
    // For the X, conversion letters ABCDEF are used.
    // 
    // cf. cppreference.com
    // int 
    //   snprintf(char* buffer, std::size_t buf_size, const char* format, ...);
    // Writes results to character string buffer. At most buf_size - 1
    // characters written. Resulting character string will be terminated with 
    // null character, unless buf_size is 0. If buf_size is zero, nothing is
    // written and buffer may be null ptr; however return value (number of bytes)
    // that would've been written) is still calculated and returned.
    char buffer[4];

    snprintf(buffer, 4, "%01x", x_as_chars[i]);

    result += std::string{buffer};
  }
  return result;
}

template <typename T>
void ToBytes<T>::decreasing_addresses_print() const
{
  constexpr std::size_t number_of_bytes {sizeof(T)};

  auto x_as_chars = reinterpret_cast<const unsigned char*>(&x_);

  for (std::size_t i {number_of_bytes - 1}; i > 0; --i)
  {
    printf("%01x ", x_as_chars[i]);
  }
  printf("%01x ", x_as_chars[0]);
}

template <typename T>
std::string ToBytes<T>::decreasing_addresses_hex() const
{
  constexpr std::size_t number_of_bytes {sizeof(T)};

  auto x_as_chars = reinterpret_cast<const unsigned char*>(&x_);

  std::string result;

  for (std::size_t i {number_of_bytes - 1}; i > 0; --i)
  {
    char buffer[4];

    snprintf(buffer, 4, "%01x", x_as_chars[i]);

    result += std::string{buffer};
  }

  char buffer[4];

  snprintf(buffer, 4, "%01x", x_as_chars[0]);

  result += std::string{buffer};

  return result;
}

template <typename T>
void ToBytes<T>::set(T& x)
{
  x_ = x;
}

} // namespace Utilities

#endif // UTILITIES_TO_BYTES_H