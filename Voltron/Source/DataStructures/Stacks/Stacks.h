//------------------------------------------------------------------------------
/// \file Stack.h
/// \author Ernest Yeung
/// \brief Classes and functions demonstrating stack.
/// \details
///
/// \ref https://www.geeksforgeeks.org/stack-data-structure-introduction-program/
/// \ref https://www.techiedelight.com/stack-implementation-in-cpp/
//-----------------------------------------------------------------------------
#ifndef DATA_STRUCTURES_STACKS_STACKS_H
#define DATA_STRUCTURES_STACKS_STACKS_H

#include "DataStructures/Arrays.h"

#include <cstddef> // std::size_t
#include <memory>
#include <vector>

namespace DataStructures
{
namespace Stacks
{

// https://leetcode.com/explore/learn/card/queue-stack/230/usage-stack/1369/
template <typename T>
class StackWithVector
{
  public:

    StackWithVector() = default;

    // Insert an element into the stack.
    void push(T x)
    {
      data_.push_back(x);
    }

    // Checks whether the queue is empty or not.
    bool is_empty() const
    {
      return data_.empty();
    }

    // Get the top item from the queue.
    T top()
    {
      return data_.back();
    }    

    // Delete an element from the queue. Return true if the operation is
    // successful.
    bool pop()
    {
      if (is_empty())
      {
        return false;
      }

      data_.pop_back();
      return true;
    }

  private:

    // Store elements.
    std::vector<T> data_;
};

template <typename T>
class StackWithArray
{
  public:

    StackWithArray():
      data_{8}
    {}

    void push(T x)
    {
      data_.append(x);
    }

    bool is_empty() const
    {
      return (data_.length() == 0);
    }

    T top() const
    {
      if (!is_empty())
      {
        return data_.back();
      }
      else
      {
        return static_cast<T>(-1);
      }
    }

    bool pop()
    {
      if (is_empty())
      {
        return false;
      }
      data_.pop_back();
      return true;
    }

    std::size_t size() const
    {
      return data_.length();
    }

  private:

    Arrays::Array<T> data_;
};

} // namespace Stacks
} // namespace DataStructures

#endif // DATA_STRUCTURES_STACKS_STACKS_H