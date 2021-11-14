//------------------------------------------------------------------------------
/// \file Queue.h
/// \author Ernest Yeung
/// \brief Classes and functions demonstrating queue.
/// \details
///
/// \ref https://leetcode.com/explore/learn/card/queue-stack/228/first-in-first-out-data-structure/1366/
/// \ref https://www.techiedelight.com/stack-implementation-in-cpp/
/// \ref Sedgewick and Wayne, Algorithms, 4th. Ed., 2011, pp. 121, Sec. 1.3
/// Bags, Queues, and Stacks.
//-----------------------------------------------------------------------------
#ifndef DATA_STRUCTURES_QUEUES_QUEUE_H
#define DATA_STRUCTURES_QUEUES_QUEUE_H

#include <cstddef> // std::size_t
#include <memory>

namespace DataStructures
{
namespace Queues
{

namespace AsHierarchy
{

template <typename Item>
class Queue
{
  public:

    //--------------------------------------------------------------------------
    /// \brief Default ctor, creating an empty queue.
    //--------------------------------------------------------------------------
    Queue() = default;

    //--------------------------------------------------------------------------
    /// \brief Add an item.
    //--------------------------------------------------------------------------
    virtual void enqueue(const Item& item) = 0;

    //--------------------------------------------------------------------------
    /// \brief Remove the least recently added item.
    //--------------------------------------------------------------------------
    virtual Item dequeue() = 0;

    //--------------------------------------------------------------------------
    /// \brief Is the queue empty?
    //--------------------------------------------------------------------------
    virtual bool is_empty() const = 0;

    //--------------------------------------------------------------------------
    /// \brief Number of items in the queue.
    //--------------------------------------------------------------------------
    virtual std::size_t size() const = 0;
};

} // namespace AsHierarchy

namespace CRTP
{

template <typename Item, typename Implementation>
class Queue
{
  public:

    void enqueue(const Item item)
    {
      object()->enqueue(item);
    }

    Item dequeue()
    {
      return object()->dequeue();
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
class BaseQueueImplementation
{
  public:

    BaseQueueImplementation() = default;

    virtual ~BaseQueueImplementation() = default;

    void enqueue(const Item item) = 0;
    Item dequeue() = 0;
    bool is_empty() const = 0;
    std::size_t size() const = 0;
};

template <typename Item>
class Queue
{
  public:

    Queue() = default;

    virtual ~Queue() = default;

    void enqueue(const Item item)
    {
      implementation_->enqueue(item);
    }

    Item dequeue()
    {
      return implementation_->dequeue();
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

    std::unique_ptr<BaseQueueImplementation<Item>> implementation_;
};

} // namespace Pimpl

} // namespace Queues
} // namespace DataStructures

#endif // DATA_STRUCTURES_QUEUES_QUEUE_H