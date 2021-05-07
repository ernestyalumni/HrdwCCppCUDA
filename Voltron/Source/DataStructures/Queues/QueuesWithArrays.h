#ifndef DATA_STRUCTURES_QUEUES_QUEUES_WITH_ARRAYS_H
#define DATA_STRUCTURES_QUEUES_QUEUES_WITH_ARRAYS_H

#include "QueuesImplementations.h"

#include "DataStructures/Arrays/FixedSizeArrays.h"

#include <cstddef> // std::size_t

namespace DataStructures
{
namespace Queues
{

namespace CRTP
{

template <typename T, std::size_t N>
class QueueWithHeadTailFixedSizeArrayOnStack :
  public QueueWithHeadTail<T, QueueWithHeadTailFixedSizeArrayOnStack<T, N>>
{
  public:

    using size_t = std::size_t;

    QueueWithHeadTailFixedSizeArrayOnStack() = default;

    const size_t head() const
    {
      return head_;
    }

    void head(const size_t index)
    {
      head_ = index;
    }

    const size_t tail() const
    {
      return tail_;
    }

    void tail(const size_t index)
    {
      tail_ = index;
    }

    T& operator[](const size_t index)
    {
      return array_[index];
    }

    const T& operator[](const size_t index) const
    {
      return array_[index];
    }

    const size_t length() const
    {
      return N;
    }

  private:

    DataStructures::Arrays::FixedSizeArrayOnStack<T, N> array_;

    size_t head_ {0};
    size_t tail_ {0};
};

} // namespace CRTP

} // namespace Queues
} // namespace DataStructures

#endif // DATA_STRUCTURES_QUEUES_QUEUES_WITH_ARRAYS_H
