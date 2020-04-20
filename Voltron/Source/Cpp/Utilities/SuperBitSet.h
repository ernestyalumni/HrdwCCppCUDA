//------------------------------------------------------------------------------
/// \file SuperBitSet.h
/// \author Ernest Yeung
/// \brief std::bitset extension.
/// \ref https://en.cppreference.com/w/cpp/utility/bitset
///-----------------------------------------------------------------------------
#ifndef CPP_UTILITIES_BIT_SET_H
#define CPP_UTILITIES_BIT_SET_H

#include <bitset>
// cf. https://en.cppreference.com/w/cpp/types/climits
#include <climits> // CHAR_BIT

namespace Cpp
{
namespace Utilities
{

namespace
{

//constexpr std::size_t number_of_bits_in_a_byte = 8;
constexpr std::size_t number_of_bits_in_a_byte = CHAR_BIT;

} // anonymous namespace

template <std::size_t N = sizeof(unsigned long long) * number_of_bits_in_a_byte>
class SuperBitSet : public std::bitset<N>
{
  public:
    // cf. cppreference.com
    // Includes the following constructors from std::bitset<N>::bitset
    //
    // constexpr bitset() noexcept;
    // Default ctor; constructs a bitset with all bits set to 0
    //
    // constexpr bitset(unsigned long long val) noexcept;
    // Constructs bitset, initializing 1st (rightmost, least significant) M bit
    // positions to corresponding bit values of val, where M is smaller of number
    // of bits in unsigned long long and number of bits N in bitset being
    // constructed.
    // 
    // explicit bitset(const std::basic_string<CharT,Traits,Alloc>& str,
    //   typename std::basic_string<CharT,Traits,Alloc>::size_type pos = 0);
    // Constructs bitset using characters in std::basic_string str. An optional
    // starting position pos and length n can be provided, as well as characters
    // denoting alternative values for set (one) and unset (0) bits.
    using std::bitset<N>::bitset;

    // TODO: Consider adding Copy construction.

    // Bit inversion.
    //SuperBitSet operator~() const
    //{
    //  const std::bitset<N>& bits {*this};

    //  return ~bits;
    //}

    // Insert bits to the "front", 0th position
    void insert_at_0(const bool value)
    {
      *this <<= 1;
      (*this)[0] = value;
    }

    // Insert multiple 0 bits at "front"
    void insert_zeroes_from_0(const std::size_t number_of_zeroes)
    {
      *this <<= number_of_zeroes;
    }

    // Remove bit(s) starting from "front"
    void remove_from_0(const std::size_t number_of_bits)
    {
      *this >>= number_of_bits;
    }
};

template <std::size_t N = sizeof(unsigned long long) * number_of_bits_in_a_byte>
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
} // namespace Cpp

#endif // CPP_UTILITIES_BIT_SET_H