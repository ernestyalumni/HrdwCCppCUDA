#ifndef DATA_STRUCTURES_STACKS_STACKS_IMPLEMENTATIONS_H
#define DATA_STRUCTURES_STACKS_STACKS_IMPLEMENTATIONS_H

#include "Stacks.h"

#include "DataStructures/Arrays/ResizeableArray.h"

#include <cstddef> // std::size_t
#include <stdexcept> // std::runtime_error

namespace DataStructures
{
namespace Stacks
{

namespace CRTP
{

template <typename Item>
class StackAsResizeableArray : public Stack<Item, StackAsResizeableArray<Item>>
{
  public:

    using ItemArray = DataStructures::Arrays::CRTP::ResizeableArray<Item>;

    StackAsResizeableArray() = default;

    void push(const Item item)
    {
      data_.append(item);
    }

    Item pop()
    {
      if (is_empty())
      {
        std::runtime_error("StackAsArray: stack is empty when popping.");
      }

      // O(1) complexity, but require keeping track with a std::size_t counter, 
      // size_.
      return data_.pop();
    }

    bool is_empty() const
    {
      return (size() <= 0);
    }

    std::size_t size() const
    {
      return data_.size();
    }

  private:

    ItemArray data_;
};

} // namespace CRTP

namespace Pimpl
{

template <typename Item>
class ArrayStackImplementation : BaseStackImplementation<Item>
{
  public:

    using ItemArray = DataStructures::Arrays::CRTP::ResizeableArray<Item>;

    ArrayStackImplementation() = default;

    void push(const Item item)
    {
      data_.append(item);
    }

    Item pop()
    {
      if (is_empty())
      {
        std::runtime_error("QueueAsArray: queue is empty when dequeuing.");
      }

      // O(1) complexity, but require keeping track with a std::size_t counter, 
      // size_.
      return data_.pop();
    }

    bool is_empty() const
    {
      return (size() <= 0);
    }

    std::size_t size() const
    {
      return data_.size();
    }

  private:

    ItemArray data_;
};

} // namespace Pimpl

} // namespace Stacks
} // namespace DataStructures

#endif // DATA_STRUCTURES_STACKS_STACKS_IMPLEMENTATIONS_H