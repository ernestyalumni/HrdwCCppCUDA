//------------------------------------------------------------------------------
/// \file BitSet.h
/// \author Ernest Yeung
/// \brief std::bitset extension.
/// \ref https://en.cppreference.com/w/cpp/utility/bitset
///-----------------------------------------------------------------------------
#ifndef _UTILITIES_BIT_SET_H_
#define _UTILITIES_BIT_SET_H_

#include <bitset>

namespace Utilities
{

constexpr std::size_t number_of_bits_in_a_byte = 8;

template <std::size_t N = sizeof(unsigned long) * number_of_bits_in_a_byte>
class BitSet
{
  public:

    // default constructors
    BitSet();

    template <typename T>
    BitSet(T x);

    // cf. https://en.cppreference.com/w/cpp/language/operator_assignment    
    template <typename T>
    void operator=(const T& x);

  private:

    std::bitset<N> bitset_;
};

template <std::size_t N>
BitSet<N>::BitSet():
  bitset_{}
{}

template <std::size_t N>
template <typename T>
BitSet<N>::BitSet(T x):
  bitset_{x}
{}

template <std::size_t N>
template <typename T>
void BitSet<N>::operator=(const T& x)
{
  bitset_ = x;
}

template <typename T>
class BitSet2 : public std::bitset<sizeof(T) * number_of_bits_in_a_byte>
{
  using BitSetT = std::bitset<sizeof(T) * number_of_bits_in_a_byte>;

  // cf. https://en.cppreference.com/w/cpp/utility/bitset/bitset
  // Use the std::bitset<N> default constructor: constructs a bitset with all
  // bits set to 0.
  // Also inherits
  // constexpr bitset(unsigned long long val) noexcept; where it
  // constructs a bitset, initializing 1st (rightmost, least significant) M bit
  // positions to corresponding bit N in bitset being constructed.
  using BitSetT::BitSetT; 
};


} // namespace Utilities

#endif // _UTILITIES_BIT_SET_H_