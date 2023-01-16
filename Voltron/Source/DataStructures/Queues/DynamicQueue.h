//------------------------------------------------------------------------------
/// \ref https://ece.uwaterloo.ca/~dwharder/aads/Projects/2/Dynamic_queue/src/
//-----------------------------------------------------------------------------
#ifndef DATA_STRUCTURES_QUEUES_DYNAMIC_QUEUE_H
#define DATA_STRUCTURES_QUEUES_DYNAMIC_QUEUE_H

#include "Queue.h"

#include <algorithm>
#include <cassert>
#include <cstddef> // std::size_t
#include <optional>
#include <stdexcept>

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

    constexpr static std::size_t default_size_ {10};

    //--------------------------------------------------------------------------
    /// \brief Default ctor, creating an empty queue.
    //--------------------------------------------------------------------------
    explicit DynamicQueue():
      size_{0},
      front_{0},
      back_{std::nullopt},
      array_capacity_{default_size_},
      array_{new T[default_size_]}
    {}

    explicit DynamicQueue(const std::size_t N):
      size_{0},
      front_{0},
      back_{std::nullopt},
      array_capacity_{std::max(N, static_cast<std::size_t>(1))},
      array_{new T[N]}
    {}

    //--------------------------------------------------------------------------
    /// \brief Copy constructor.
    //--------------------------------------------------------------------------
    DynamicQueue(const DynamicQueue&);

    //--------------------------------------------------------------------------
    /// \brief Copy assignment.
    //--------------------------------------------------------------------------
    DynamicQueue& operator=(const DynamicQueue&);

    DynamicQueue(DynamicQueue&&) = delete;
    DynamicQueue& operator=(DynamicQueue&&) = delete;

    virtual ~DynamicQueue()
    {
      delete [] array_;
    }

    T front() const
    {
      if (is_empty())
      {
        throw std::runtime_error("Called front on empty DynamicQueue");
      }

      return array_[front_];
    }

    T head() const
    {
      return front();
    }

    //--------------------------------------------------------------------------
    /// \brief Add an item.
    //--------------------------------------------------------------------------
    virtual void enqueue(const T& item) override;

    //--------------------------------------------------------------------------
    /// \brief Remove the least recently added item.
    //--------------------------------------------------------------------------
    virtual T dequeue() override
    {
      if (is_empty())
      {
        throw std::runtime_error("Called dequeue on an empty DynamicQueue");
      }

      if (front_ == *back_)
      {
        back_ = std::nullopt;
        
        const std::size_t old_front_ = front_;
        front_ = 0;

        size_ = 0;
        return array_[old_front_];
      }

      const std::size_t old_front_ {front_};

      front_ = (front_ + 1) % array_capacity_;

      --size_;

      return array_[old_front_];
    }

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
    virtual bool is_empty() const override;

    //--------------------------------------------------------------------------
    /// \brief Number of items in the queue.
    //--------------------------------------------------------------------------
    virtual std::size_t size() const override;

  protected:

    bool is_full()
    {
      return back_.has_value() ?
        (((*back_) + 1) % array_capacity_ == front_) :
        false;
    }

    T back() const
    {
      assert(!is_empty());

      return array_[*back_];
    }

    std::size_t get_front_index() const
    {
      return front_;
    }

    std::size_t get_back_index() const
    {
      if (!back_.has_value())
      {
        throw std::runtime_error(
          "Failed to get value for back index in DynamicQueue");
      }

      return *back_;
    }

    std::size_t get_size() const
    {
      return size_;
    }

    std::size_t get_array_capacity() const
    {
      return array_capacity_;
    }

  private:

    void double_capacity()
    {
      assert(is_full());

      T* tmp_array = new T[2 * array_capacity_];

      if (front_ <= *back_)
      {
        assert(front_ == 0 && *back_ == array_capacity_ - 1);

        for (std::size_t i {0}; i < (*back_ + 1 - front_); ++i)
        {
          tmp_array[i] = array_[i + front_];
        }
      }
      else
      {
        for (std::size_t i {0}; i < (array_capacity_ - front_); ++i)
        {
          tmp_array[i] = array_[i + front_];
        }

        for (std::size_t i {0}; i < front_; ++i)
        {
          tmp_array[i + (array_capacity_ - front_)] = array_[i];
        }
      }

      delete [] array_;

      array_ = tmp_array;

      // Reset the front and back pointers.
      front_ = 0;
      back_ = array_capacity_ - 1;

      array_capacity_ *= 2;
    }

    std::size_t size_;
    std::size_t front_;
    //--------------------------------------------------------------------------
    /// This gives the very last element's index of the current elements in the
    /// array, and not the index for the very next element.
    //--------------------------------------------------------------------------
    std::optional<std::size_t> back_;
    std::size_t array_capacity_;
    T* array_;
};

template <typename T>
DynamicQueue<T>::DynamicQueue(const DynamicQueue& other):
  size_{other.size_},
  front_{other.front_},
  back_{other.back_},
  array_capacity_{other.array_capacity_},
  array_{new T[array_capacity_]}
{
  std::copy(
    std::begin(other.array_),
    std::end(other.array_),
    std::begin(array_));
}

template <typename T>
bool DynamicQueue<T>::is_empty() const
{
  return !(back_.has_value());
}

template <typename T>
void DynamicQueue<T>::enqueue(const T& item)
{
  if (is_full())
  {
    double_capacity();
  }

  back_ = is_empty() ? 0 : (*back_ + 1) % array_capacity_;

  array_[*back_] = item;

  ++size_;
}

template <typename T>
std::size_t DynamicQueue<T>::size() const
{
  return size_;
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