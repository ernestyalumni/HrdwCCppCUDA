#ifndef DATA_STRUCTURES_QUEUES_QUEUES_IMPLEMENTATIONS_H
#define DATA_STRUCTURES_QUEUES_QUEUES_IMPLEMENTATIONS_H

#include "Queues.h"

#include "DataStructures/Arrays/ResizeableArray.h"

#include <cstddef> // std::size_t
#include <stdexcept> // std::runtime_error

namespace DataStructures
{
namespace Queues
{

namespace CRTP
{

template <typename Item>
class QueueAsArray : public Queue<Item, QueueAsArray<Item>>
{
  public:

    using ItemArray = DataStructures::Arrays::CRTP::ResizeableArray<Item>;

    QueueAsArray() = default;

    void enqueue(const Item item)
    {
      data_.append(item);
    }

    Item dequeue()
    {
      if (is_empty())
      {
        std::runtime_error("QueueAsArray: queue is empty when dequeuing.");
      }

      // O(1) complexity.
      Item return_value {data_.get_value(0)};

      // O(N) complexity.
      for (std::size_t i {0}; i < (size() - 1); ++i)
      {
        data_[i] = data_[i + 1];
      }

      data_.pop();

      return return_value;
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


template <typename Item, class Implementation>
class QueueWithHeadTail :
  public Queue<Item, QueueWithHeadTail<Item, Implementation>>
{
  public:

    using size_t = std::size_t;

    const size_t head() const
    {
      return object().head();
    }

    void head(const size_t index)
    {
      object()->head(index);
    }

    const size_t tail() const
    {
      return object().tail();
    }

    void tail(const size_t index)
    {
      object()->tail(index);
    }   

    const size_t length() const
    {
      return object().length();
    }

    //--------------------------------------------------------------------------
    /// \ref pp. 235, Sec. 10.1 Stacks and queues, Introduction to Algorithms.
    /// \details
    /// Time Complexity: O(1)
    /// Space Complexity: O(N), N = array length, O(2) for actual operation.
    //--------------------------------------------------------------------------
    void enqueue(const Item item)
    {
      const size_t tail_index {tail()};

      object().operator[](tail_index) = item;

      if (tail_index == (object().length() - 1))
      {
        object().tail(0);
      }
      else
      {
        object().tail(tail_index + 1);
      }
    }

    //--------------------------------------------------------------------------
    /// \ref pp. 235, Sec. 10.1 Stacks and queues, Introduction to Algorithms.
    /// \details
    /// Time Complexity: O(1)
    /// Space Complexity: O(N), N = array length, O(2) for actual operation.
    //--------------------------------------------------------------------------
    Item dequeue()
    {
      const size_t head_index {head()};
      Item x {object().operator[](head_index)};

      if (head_index == (length() - 1))
      {
        object().head(0);
      }
      else
      {
        object().head(head_index + 1);
      }

      return x;
    }

    bool is_empty() const
    {
      return (tail() == head());
    }

    std::size_t size() const
    {
      const size_t head_index {head()};
      const size_t tail_index {tail()};

      if (head_index <= tail_index)
      {
        return (tail_index - head_index);
      }
      else
      {
        return (length() - head_index) + (tail_index);
      }
    }

  
  private:

    Implementation& object()
    {
      return static_cast<Implementation&>(*this);
    }

    const Implementation& object() const
    {
      return static_cast<const Implementation&>(*this);
    }
};

} // namespace CRTP

} // namespace Queues
} // namespace DataStructures

#endif // DATA_STRUCTURES_QUEUES_QUEUES_IMPLEMENTATIONS_H