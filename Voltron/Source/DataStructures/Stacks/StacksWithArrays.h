#ifndef DATA_STRUCTURES_STACKS_STACKS_WITH_ARRAYS_H
#define DATA_STRUCTURES_STACKS_STACKS_WITH_ARRAYS_H

#include "DataStructures/Arrays/FixedSizeArrays.h"

#include <cstddef> // std::size_t
#include <optional>
#include <stdexcept> // std::runtime_error

namespace DataStructures
{
namespace Stacks
{

//------------------------------------------------------------------------------
/// \ref Cormen, Leiserson, Rivest, and Stein (2009), pp. 235. Exercises 10.1-2
/// Push and Pop operations should run in O(1) time.
//------------------------------------------------------------------------------
template <typename T, std::size_t N>
class TwoStacksOneArray
{
  public:

    template <typename U>
    using optional = std::optional<U>;

    using size_t = std::size_t;
    using Array = DataStructures::Arrays::FixedSizeArrayOnStack<T, N>;

    TwoStacksOneArray() = default;

    void push_1(const T t)
    {
      if ((top_index_2_ && *top_index_2_ == 0) ||
        (top_index_1_ && *top_index_1_ == N - 1) ||
        (top_index_1_ && top_index_2_ && *top_index_1_ + 1 == *top_index_2_))
      {
        std::runtime_error("TwoStacksOneArray: overflow");        
      }

      if (!top_index_1_)
      {
        top_index_1_.emplace(0);
      }
      else
      {
        *top_index_1_ += 1;
      }

      data_[*top_index_1_] = t;
    }

    void push_2(const T t)
    {
      if ((top_index_2_ && *top_index_2_ == 0) ||
        (top_index_1_ && *top_index_1_ == N - 1) ||
        (top_index_1_ && top_index_2_ && *top_index_1_ == *top_index_2_ - 1))
      {
        std::runtime_error("TwoStacksOneArray: overflow");        
      }

      if (!top_index_2_)
      {
        top_index_2_.emplace(N - 1);
      }
      else
      {
        *top_index_2_ -= 1;
      }

      data_[*top_index_2_] = t;
    }

    T pop_1()
    {
      if (is_empty_1())
      {
        std::runtime_error("TwoStacksOneArray: underflow on 1");
      }

      const T popped {data_[*top_index_1_]};

      if (*top_index_1_ == 0)
      {
        top_index_1_.reset();
      }
      else
      {
        *top_index_1_ -= 1;
      }

      return popped;
    }

    T pop_2()
    {
      if (is_empty_2())
      {
        std::runtime_error("TwoStacksOneArray: underflow on 2");
      }

      const T popped {data_[*top_index_2_]};

      if (*top_index_2_ == N - 1)
      {
        top_index_2_.reset();
      }
      else
      {
        *top_index_2_ += 1;
      }

      return popped;
    }


    bool is_empty_1() const
    {
      return !top_index_1_;
    }

    bool is_empty_2() const
    {
      return !top_index_2_;
    }

    size_t size() const
    {
      size_t total_size {0};

      if (top_index_1_)
      {
        total_size += *top_index_1_ + 1;
      }

      if (top_index_2_)
      {
        total_size += (N - *top_index_2_);
      }

      return total_size;
    }


    T top_1() const
    {
      return data_[*top_index_1_];
    }

    T top_2() const
    {
      return data_[*top_index_2_];
    }

  protected:

    optional<size_t> top_index_1() const
    {
      return top_index_1_;
    }

    optional<size_t> top_index_2() const
    {
      return top_index_2_;
    }

  private:

    Array data_;

    optional<size_t> top_index_1_ {};
    optional<size_t> top_index_2_ {};
};

} // namespace Stacks
} // namespace DataStructures

#endif // DATA_STRUCTURES_STACKS_STACKS_WITH_ARRAYS_H