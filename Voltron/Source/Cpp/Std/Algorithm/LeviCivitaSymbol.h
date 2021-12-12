#ifndef CPP_STD_ALGORITHM_LEVI_CIVITA_SYMBOL_H
#define CPP_STD_ALGORITHM_LEVI_CIVITA_SYMBOL_H

#include <algorithm>
#include <array>
#include <cstddef>
#include <numeric>

namespace Cpp
{
namespace Std
{
namespace Algorithm
{

template <std::size_t N = 3>
class LeviCivitaSymbol
{
  public:

    LeviCivitaSymbol(const unsigned int starting_value = 0):
      eps_{},
      counter_{0},
      starting_value_{starting_value},      
      sign_{1},
      mod_2_sign_{1}
    {
      std::iota(eps_.begin(), eps_.end(), starting_value_);
    }

    void reset()
    {
      std::iota(eps_.begin(), eps_.end(), starting_value_);
    }

    std::array<unsigned int, N + 1> operator()()
    {
      std::array<unsigned int, N + 1> result;

      std::copy(eps_.cbegin(), eps_.cend(), result.begin());

      result[N] = mod_2_sign_ * sign_;

      // Take steps for next permutation.

      const bool is_not_last_permutation {
        std::next_permutation(eps_.begin(), eps_.end())};

      if (is_not_last_permutation)
      {
        counter_ += 1;
      }
      else
      {
        counter_ = 0;
      }

      mod_2_sign_ = (((counter_ / 2) % 2) == 1) ? -1 : 1;

      sign_ = (sign_ == 1) ? -1 : 1;

      return result;
    }

  private:

    std::array<unsigned int, N> eps_;
    std::size_t counter_;
    unsigned int starting_value_;
    int sign_;
    int mod_2_sign_;
};

//------------------------------------------------------------------------------
/// \ref https://stackoverflow.com/questions/213761/what-are-some-uses-of-template-template-parameters
/// \ref https://stackoverflow.com/questions/16464297/a-simple-code-to-detect-the-permutation-sign
//------------------------------------------------------------------------------

template <typename ContainerT>
bool is_even(const ContainerT& a, const std::size_t N)
{
  // Count the number of inversions (pairs of elements out of order).
  std::size_t counter {0};
  for (std::size_t i {0}; i < N; ++i)
  {
    for (std::size_t j {i + 1}; j < N; ++j)
    {
      // Out of order from ascending.
      if (a[i] > a[j])
      {
        ++counter;
      }
    }
  }

  return (counter % 2 == 0);
}

template <template <class> class ContainerT, typename ElementT>
bool is_even(const ContainerT<ElementT>& a)
{
  return is_even(a, a.size());
}

} // namespace Algorithm
} // namespace Std
} // namespace Cpp

#endif// CPP_STD_ALGORITHM_LEVI_CIVITA_SYMBOL_H