//------------------------------------------------------------------------------
/// \ref https://ece.uwaterloo.ca/~dwharder/aads/Projects/2/Dynamic_queue/src/
//-----------------------------------------------------------------------------
#ifndef DATA_STRUCTURES_QUEUES_DYNAMIC_QUEUE_H
#define DATA_STRUCTURES_QUEUES_DYNAMIC_QUEUE_H

#include "Queue.h"

#include <algorithm>
#include <cstddef> // std::size_t
#include <optional>

namespace DataStructures
{
namespace Queues
{

namespace AsHierarchy
{

//-----------------------------------------------------------------------------
/// \ref 3.03.Queues.pdf, 3.3.3.2 Two-ended/Circular Array Implementation.
/// \details A 1-ended array only allows O(1) insertions and erases at the back
/// and thus we'll restrict our operations to that part of the array, and
/// therefore cannot simulate a queue. Instead, use a 2-ended queue that'll
/// allow insertions and erases at both the front and back in O(1)
//-----------------------------------------------------------------------------


template <typename T>
class DynamicQueue : Queue<T>
{
  public:

    //--------------------------------------------------------------------------
    /// \brief Default ctor, creating an empty queue.
    //--------------------------------------------------------------------------
    DynamicQueue(const std::size_t N);

    //--------------------------------------------------------------------------
    /// \brief Copy constructor.
    //--------------------------------------------------------------------------
    DynamicQueue(const DynamicQueue&);

    //--------------------------------------------------------------------------
    /// \brief Copy assignment.
    //--------------------------------------------------------------------------
    DynamicQueue& operator=(const DynamicQueue&);

    virtual ~DynamicQueue();

    T front() const;    

    T head() const
    {
      return front();
    }

    //--------------------------------------------------------------------------
    /// \brief Add an item.
    //--------------------------------------------------------------------------
    void enqueue(const T& item);

    //--------------------------------------------------------------------------
    /// \brief Remove the least recently added item.
    //--------------------------------------------------------------------------
    T dequeue();

    void push(const T& item)
    {
      enqueue(item);
    }

    T pop()
    {
      return dequeue();
    }

    //--------------------------------------------------------------------------
    /// \brief Is the queue empty?
    //--------------------------------------------------------------------------
    bool is_empty() const;

    //--------------------------------------------------------------------------
    /// \brief Number of items in the queue.
    //--------------------------------------------------------------------------
    std::size_t size() const;

  private:

    void double_capacity()
    {

    }

    std::size_t queue_size_;
    std::size_t queue_front_;
    std::optional<std::size_t> queue_back_;
    std::size_t array_capacity_;
    T* array_;
};

template <typename T>
DynamicQueue<T>::DynamicQueue(const std::size_t N):
  queue_size_{0},
  queue_front_{0},
  queue_back_{std::nullopt},
  array_capacity_{std::max(N, 1)},
  array_{new T[N]}
{}

template <typename T>
DynamicQueue<T>::~DynamicQueue()
{
  delete [] array_;
}

template <typename T>
T DynamicQueue<T>::front() const
{
  if (is_empty())
  {
    throw std::runtime_error("Called front on empty DynamicQueue")
  }

  return array_[queue_front_];
}


template <typename T>
bool DynamicQueue<T>::is_empty() const
{
  return queue_size_ == 0;
}

template <typename T>
void DynamicQueue<T>::enqueue(const T& item)
{
  if (queue_size_ == array_capacity_)
  {
    double_capacity();
  }

  if (!(queue_back_.has_value()))
  {
    queue_back_ = 0
  }
  else
  {
    ++(*queue_back_);
  }

  array_[*queue_back_] = item;

  ++queue_size_;
}

template <typename T>
void DynamicQueue<T>::dequeue()
{
  if (is_empty())
  {
    throw std::runtime_error("Called dequeue on an empty DynamicQueue")   
  }

  --queue_size_;
  ++queue_front_;

  return array_[queue_front_ - 1];
}

} // namespace AsHierarchy

namespace CRTP
{

} // namespace CRTP

namespace Pimpl
{

} // namespace Pimpl

} // namespace Queues
} // namespace DataStructures

#endif // DATA_STRUCTURES_QUEUES_DYNAMIC_QUEUE_H