//------------------------------------------------------------------------------
/// \ref pp. 234, Ch. 10 Elementary Data Structures, Introduction to Algorithms,
/// Cormen, Leiserson, Rivest, Stein. 
/// \ref https://ece.uwaterloo.ca/~dwharder/aads/Projects/2/Dynamic_queue/src/
//-----------------------------------------------------------------------------
#ifndef DATA_STRUCTURES_QUEUES_CIRCULAR_QUEUE_H
#define DATA_STRUCTURES_QUEUES_CIRCULAR_QUEUE_H

#include "Queue.h"

#include <algorithm> // std::max
#include <cassert>
#include <cstddef> // std::size_t
#include <optional>
#include <utility> // std::move

namespace DataStructures
{
namespace Queues
{

namespace AsHierarchy
{

namespace CLRS
{

template <typename T>
class CircularQueue : Queue<T>
{
  public:

    //--------------------------------------------------------------------------
    /// \brief Default ctor, creating an empty queue.
    //--------------------------------------------------------------------------
    CircularQueue(const std::size_t N = 10):
      array_{new T[std::max(N, static_cast<std::size_t>(1))]},
      array_capacity_{std::max(N, static_cast<std::size_t>(1))},
      size_{0},
      head_{0},
      tail_{0}
    {}

    //--------------------------------------------------------------------------
    /// \brief Copy constructor.
    //--------------------------------------------------------------------------
    CircularQueue(const CircularQueue& other):
      array_{new T[other.array_capacity_]},
      array_capacity_{other.array_capacity_},
      size_{other.size_},
      head_{other.head_},
      tail_{other.tail_}
    {
      std::copy(
        other.array_,
        other.array_ + other.array_capacity_,
        array_);
    }

    //--------------------------------------------------------------------------
    /// \brief Copy assignment.
    //--------------------------------------------------------------------------
    CircularQueue& operator=(const CircularQueue& other)
    {
      delete[] array_;
      array_ = new T[other.array_capacity_];
      array_capacity_ = other.array_capacity_;
      size_ = other.size_;
      head_ = other.head_;
      tail_ = other.tail_;

      std::copy(other.array_, other.array_ + other.array_capacity_, array_);

      return *this;
    }      

    //--------------------------------------------------------------------------
    /// \brief Move constructor.
    //--------------------------------------------------------------------------
    CircularQueue(CircularQueue&& other):
      array_{other.array_},
      array_capacity_{other.array_capacity_},
      size_{other.size_},
      head_{other.head_},
      tail_{other.tail_}
    {
      // So as not to invoke delete in dtor of other.
      other.array_ = nullptr;
    }

    //--------------------------------------------------------------------------
    /// \brief Move assignment.
    //--------------------------------------------------------------------------
    CircularQueue& operator=(CircularQueue&& other)
    {
      if (array_ != nullptr)
      {
        delete [] array_;
      }

      array_ = other.array_;

      array_capacity_ = std::move(other.array_capacity_);
      size_ = std::move(other.size_);
      head_ = std::move(other.head_);
      tail_ = std::move(other.tail_);

      // So as not to invoke delete in dtor of other.
      other.array_ = nullptr;

      return *this;
    }      

    virtual ~CircularQueue()
    {
      if (array_ != nullptr)
      {
        delete [] array_;
      }
    }

    void enqueue(const T& item) override
    {
      if (!is_full())
      {
        array_[tail_] = item;
        tail_ = (tail_ + 1) % array_capacity_;
        ++size_;
      }
    }

    T dequeue() override
    {
      if (!is_empty())
      {
        const std::size_t old_head_ {head_};

        head_ = (head_ + 1) % array_capacity_;

        --size_;

        return array_[old_head_];
      }
      else
      {
        throw std::runtime_error(
          "Called dequeue on an empty CircularQueue");
      }
    }

    bool is_empty() const override
    {
      return head_ == tail_;
    }

    std::size_t size() const
    {
      return size_;
    }

    //--------------------------------------------------------------------------
    /// \details Here, it's important to know the fact that this is a queue of
    /// at most n - 1 elements using an array Q[1...n].
    /// \ref pp. 234 Ch. 10 "Elementary Data Structures" of CLRS.
    /// \ref https://stackoverflow.com/questions/16395354/why-q-head-q-tail-1-represents-the-queue-is-full-in-clrs
    //--------------------------------------------------------------------------
    bool is_full() const
    {
      return head_ == (tail_ + 1) % array_capacity_;
    }

  protected:

    std::size_t get_head() const
    {
      return head_;
    }    

    std::size_t get_tail() const
    {
      return tail_;
    }    

    std::size_t get_array_capacity() const
    {
      return array_capacity_;
    }    

    bool is_null_array() const
    {
      return array_ == nullptr;
    }

  private:

    T* array_;
    std::size_t array_capacity_;
    std::size_t size_;
    std::size_t head_;
    // Indexes the next location at which a newly arriving element will be
    // inserted into the queue.
    std::size_t tail_;
};

} // namespace CLRS

} // namespace AsHierarchy

} // namespace Queues
} // namespace DataStructures

#endif // DATA_STRUCTURES_QUEUES_DYNAMIC_QUEUE_H