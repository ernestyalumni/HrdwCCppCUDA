//------------------------------------------------------------------------------
/// \file ToBytes.h
/// \author Ernest Yeung
/// \brief .
/// \ref 
///-----------------------------------------------------------------------------
#ifndef _UTILITIES_TO_BYTES_H_
#define _UTILITIES_TO_BYTES_H_

#include <cstddef> // std::size_t
#include <cstdio> // printf

namespace Utilities
{

template <typename T>
class ToBytes
{
  public:

    ToBytes() = delete;

    ToBytes(T& x);

    void increasing_addresses_print() const;

    void decreasing_addresses_print() const;

    void set(T& x);

  private:

    T x_;
};

template <typename T>
ToBytes<T>::ToBytes(T& x):
  x_{x}
{}

template <typename T>
void ToBytes<T>::increasing_addresses_print() const
{
  constexpr std::size_t number_of_bytes {sizeof(T)};

  auto x_as_chars = reinterpret_cast<const unsigned char*>(&x_);

  for (std::size_t i {0}; i < number_of_bytes; ++i)
  {
    printf("%01x ", x_as_chars[i]);
  }
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
void ToBytes<T>::set(T& x)
{
  x_ = x;
}

} // namespace Utilities

#endif // _UTILITIES_TO_BYTES_H_