//------------------------------------------------------------------------------
/// \file Stack.h
/// \author Ernest Yeung
/// \brief Classes and functions demonstrating stack.
/// \details
///
/// \ref https://www.geeksforgeeks.org/stack-data-structure-introduction-program/
/// \ref https://www.techiedelight.com/stack-implementation-in-cpp/
//-----------------------------------------------------------------------------
#ifndef DATA_STRUCTURES_STACKS_STACK_H
#define DATA_STRUCTURES_STACKS_STACK_H

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
      return data_back();
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
class Stack
{
  public:


    T peek()
    {
      if (top_ < 0)
      {

      }
    }

  private:

    long top_;
};

} // namespace Stacks
} // namespace DataStructures

#endif // DATA_STRUCTURES_STACKS_STACK_H