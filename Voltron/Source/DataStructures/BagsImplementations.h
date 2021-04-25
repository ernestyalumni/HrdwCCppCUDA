#ifndef DATA_STRUCTURES_BAGS_IMPLEMENTATION_H
#define DATA_STRUCTURES_BAGS_IMPLEMENTATION_H

#include "Bags.h"
#include "Iterable.h"

#include <cmath>
#include <cstddef> // std::size_t
#include <initializer_list>
#include <iostream>
#include <utility>
#include <vector>

namespace DataStructures
{
namespace Bags
{

template <typename Item>
class BagAsVector :
  public Bag<Item>,
  public DataStructures::Iterables::Iterable<Item>
{
  public:

    BagAsVector():
      data_{}
    {}

    virtual ~BagAsVector() = default;

    Item* begin()
    {
      return data_.data();
    }

    Item* end()
    {
      return data_.data() + data_.size();
    }

    void add(const Item item)
    {
      data_.emplace_back(item);
    }

    bool is_empty() const
    {
      return data_.size() == 0;
    }

    std::size_t size() const
    {
      return data_.size();
    }

  private:

    std::vector<Item> data_;
};

//------------------------------------------------------------------------------
/// \ref Sedgewick and Wayne, Algorithms, 4th Ed. 2011, pp. 125 Sec. 1.3 Bags,
/// Queues, and Stacks
//------------------------------------------------------------------------------ 
template <template <typename Item> class BagTemplate>
double bag_statistics()
{
  BagTemplate<double> numbers;

  double number;

  //----------------------------------------------------------------------------
  /// \ref https://stackoverflow.com/questions/38978266/how-can-stdcin-return-a-bool-and-itself-at-the-same-time
  /// \details cin is of type std::basic_istream, and basic_istream has a bool.
  ///
  /// std::basic_ios<CharT,Traits>::operator bool
  /// explicit operator bool() const;
  /// Returns true if stream has no errors and ready for I/O operations.
  /// Specifically, returns !fail().
  //----------------------------------------------------------------------------
  while (std::cin >> number)
  {
    numbers.add(number);
  }

  const std::size_t N {numbers.size()};

  double sum {0.0};
  for (double x : numbers)
  {
    sum += x;
  }
  const double mean {sum/N};

  sum = 0.0;
  for (double x : numbers)
  {
    sum += (x - mean) * (x - mean);
  }
  const double standard_deviation {sqrt(sum / (N - 1.0))};

  std::cout << "Mean: " << mean << "\n";
  std::cout << "Std. dev.: " << standard_deviation << "\n";

  return mean;  
}

template <template <typename Item> class BagTemplate>
std::pair<double, double> bag_statistics(
  const std::initializer_list<double> input)
{
  BagTemplate<double> numbers;

  for (double x : input)
  {
    numbers.add(x);
  }

  const std::size_t N {numbers.size()};

  double sum {0.0};
  for (double x : numbers)
  {
    sum += x;
  }

  double mean {sum / N};

  sum = 0.0;
  for (double x : numbers)
  {
    sum += (x - mean) * (x - mean);
  }

  double standard_deviation {std::sqrt(sum / (N - 1))};

  return std::make_pair<double, double>(
    std::move(mean),
    std::move(standard_deviation));  
}

} // namespace Bags
} // namespace DataStructures

#endif // DATA_STRUCTURES_BAGS_IMPLEMENTATION_H