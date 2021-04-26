#ifndef DATA_STRUCTURES_QUEUES_IMPLEMENTATIONS_H
#define DATA_STRUCTURES_QUEUES_IMPLEMENTATIONS_H

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

} // namespace CRTP

} // namespace Queues
} // namespace DataStructures

#endif // DATA_STRUCTURES_QUEUES_IMPLEMENTATIONS_H