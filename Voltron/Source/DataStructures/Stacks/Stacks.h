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

#include "DataStructures/Arrays.h"

#include <cstddef> // std::size_t
#include <memory>
#include <vector>

namespace DataStructures
{
namespace Stacks
{

template <typename Item>
class Stack
{
  public:

    //--------------------------------------------------------------------------
    /// \brief Default ctor, creating an empty stack.
    //--------------------------------------------------------------------------
    Stack() = default;

    //--------------------------------------------------------------------------
    /// \brief Add an item.
    //--------------------------------------------------------------------------
    void push(const Item item) = 0;

    //--------------------------------------------------------------------------
    /// \brief Remove the most recently added item.
    //--------------------------------------------------------------------------
    Item pop() = 0;

    //--------------------------------------------------------------------------
    /// \brief Is the queue empty?
    //--------------------------------------------------------------------------
    bool is_empty() = 0;

    //--------------------------------------------------------------------------
    /// \brief Number of items in the queue.
    //--------------------------------------------------------------------------
    std::size_t size() const = 0;    
};

namespace CRTP
{

template <typename Item, typename Implementation>
class Stack
{
  public:

    void push(const Item item)
    {
      object()->push(item);
    }

    Item pop()
    {
      return object()->pop();
    }

    bool is_empty() const
    {
      return object()->is_empty();
    }

    std::size_t size() const
    {
      return object()->size();
    }

  private:

    Implementation& object()
    {
      return static_cast<Implementation&>(*this);
    }
};

} // namespace CRTP

namespace Pimpl
{

template <typename Item>
class BaseStackImplementation
{
  public:

    BaseStackImplementation() = default;

    virtual ~BaseStackImplementation() = default;

    void push(const Item item) = 0;
    Item pop() = 0;
    bool is_empty() const = 0;
    std::size_t size() const = 0;
};

template <typename Item>
class Stack
{
  public:

    Stack() = default;

    virtual ~Stack() = default;

    void push(const Item item)
    {
      implementation_->push(item);
    }

    Item pop()
    {
      return implementation_->pop();
    }

    bool is_empty() const
    {
      return implementation_->is_empty();
    }

    std::size_t size() const
    {
      return implementation_->size();
    }

  private:

    std::unique_ptr<BaseStackImplementation<Item>> implementation_;
};

} // namespace Pimpl

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

#endif // DATA_STRUCTURES_STACKS_STACK_H