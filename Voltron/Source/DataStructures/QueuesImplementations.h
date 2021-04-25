#ifndef DATA_STRUCTURES_QUEUES_IMPLEMENTATIONS_H
#define DATA_STRUCTURES_QUEUES_IMPLEMENTATIONS_H

#include "Queues.h"

#include "DataStructures/Arrays/ResizeableArray.h"

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

    using ItemArray = DataStructures::Arrays::ResizeableArray<Item>;

  private:

    ItemArray array_;
};

} // namespace CRTP

} // namespace Queues
} // namespace DataStructures

#endif // DATA_STRUCTURES_QUEUES_IMPLEMENTATIONS_H