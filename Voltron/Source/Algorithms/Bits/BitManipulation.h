#ifndef ALGORITHMS_BITS_BIT_MANIPULATION_H
#define ALGORITHMS_BITS_BIT_MANIPULATION_H

#include <cstddef>
#include <type_traits>

namespace Algorithms
{
namespace Bits
{

class BitManipulation
{
  public:

    //--------------------------------------------------------------------------
    /// Is ith bit set or not.
    //--------------------------------------------------------------------------
    template <typename T>
    static bool check_bit(const T x, const T i)
    {
      return ((x & (1u << i)) == 0);
    }

    //--------------------------------------------------------------------------
    /// Is ith bit set or not.
    //--------------------------------------------------------------------------
    template <typename T>
    static void set_bit_high(T& x, const T i)
    {
      x |= (1u << i);
    }

    template <typename T>
    static void clear_bit(T& x, const T i)
    {
      x &= ~(1u << i);
    }

    template <typename T>
    static void toggle_bit(T& x, const T i)
    {
      x ^= (1u << i);
    }

    template <typename T>
    static std::size_t count_1_bits(const T x)
    {
      std::size_t count {0};
      // e.g. 1100, 1011
      // If x has least significant bit (lsb) be 0, then (x-1) would have lsb be
      // 1, and the "above" bits would be 1's, 11..1, up until the bit it
      // "carried" from. 0,1 maps to 0 by &. Only the next significant bit will
      // be nonzero
      while (x > 0)
      {
        x &= (x - 1u);
        count++;
      }

      return count;
    }

    //--------------------------------------------------------------------------
    /// For some integer with some bit values, 1010, consider the two's
    /// complement: 11110101 + 1u = 11110110. The +1 would carry over until the
    /// position of the right most bit (recall that the right most bit was
    /// flipped to 0 in taking the two's complement). 
    //--------------------------------------------------------------------------
    template <typename T>
    static int get_rightmost_bit(const T x)
    {
      int input_value {static_cast<int>(x)};

      return (input_value & (-input_value))
    }
};

//------------------------------------------------------------------------------
/// \details (x & (1 << c)) != 0
//------------------------------------------------------------------------------

template <typename T>
bool is_bit_set_high(const T x, const std::size_t position)
{
  return (x & (1 << position)) != 0;
}

template <typename T>
T set_bit_high(const T x, const std::size_t position)
{
  return (x | (1 << position));
}

// cf. https://youtu.be/NLKQEOgBAnw?t=509
// Algorithms: Bit Manipulation. HackerRank.
// & (And) it with a mask with 1 but that spot (position).
template <typename T>
T clear_bit(const T x, const std::size_t position)
{
  // ~ inverts
  return (x & ~(1 << position));
}

} // namespace Bits
} // namespace Algorithms

#endif // ALGORITHMS_BITS_SHIFT_H
