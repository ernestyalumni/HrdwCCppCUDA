//------------------------------------------------------------------------------
/// \ref https://ece.uwaterloo.ca/~dwharder/aads/Projects/2/Resizable_deque/src/
//-----------------------------------------------------------------------------
#ifndef DATA_STRUCTURES_QUEUES_RESIZABLE_DEQUE_H
#define DATA_STRUCTURES_QUEUES_RESIZABLE_DEQUE_H

#include "Deque.h"

#include <algorithm>
#include <cassert>
#include <cstddef> // std::size_t
#include <optional>

namespace DataStructures
{
namespace Deques
{

namespace AsHierarchy
{

//-----------------------------------------------------------------------------
/// \ref 3.4.3.2 Circular Array Implementation. 3.04.Deques.pdf, 2011 D.W.
/// Harder, ECE 250 Algorithms and Data Structure.
//-----------------------------------------------------------------------------
template <typename T>
class ResizableDeque : Deque<T>
{
  public:

    //--------------------------------------------------------------------------
    /// \brief Default ctor, creating an empty queue.
    //--------------------------------------------------------------------------
    ResizableDeque(const std::size_t N = 10);

    //--------------------------------------------------------------------------
    /// \brief Copy constructor.
    /// \details O(N).
    //--------------------------------------------------------------------------
    ResizableDeque(const ResizableDeque&);

    //--------------------------------------------------------------------------
    /// \brief Copy assignment.
    /// \details O(max(N_lhs, N_rhs))
    //--------------------------------------------------------------------------
    ResizableDeque& operator=(const ResizableDeque&);

    virtual ~ResizableDeque();

    //--------------------------------------------------------------------------
    /// \details O(1)
    //--------------------------------------------------------------------------
    virtual T front() const override;

    //--------------------------------------------------------------------------
    /// \details O(1)
    //--------------------------------------------------------------------------
    virtual T back() const override
    {
      assert(!is_empty());

      return array_[*back_];
    }

    //--------------------------------------------------------------------------
    /// \brief Add an item.
    /// \details O(1) on average
    //--------------------------------------------------------------------------
    virtual void push_front(const T& item) override
    {
      if (is_full())
      {
        double_capacity();
      }

      if (is_empty())
      {
        front_ = 0;
        back_ = 0;
      }
      else
      {
        front_ = *front_ == 0 ? array_capacity_ - 1 : *front_ - 1; 
      }

      array_[*front_] = item;

      ++size_;
    }

    //--------------------------------------------------------------------------
    /// \details O(1) on average
    //--------------------------------------------------------------------------
    virtual void push_back(const T& item) override;

    //--------------------------------------------------------------------------
    /// \details O(1) on average
    //--------------------------------------------------------------------------
    virtual T pop_front() override;

    //--------------------------------------------------------------------------
    /// \details O(1) on average
    //--------------------------------------------------------------------------
    virtual T pop_back() override
    {
      if (is_empty())
      {
        throw std::runtime_error("Called pop_back on an empty ResizableDeque");
      }

      if (*front_ == *back_)
      {
        assert(size() == 1);

        front_ = std::nullopt;
        
        const std::size_t old_back_ = *back_;
        back_ = std::nullopt;

        size_ = 0;
        return array_[old_back_];
      }

      const std::size_t old_back_ = *back_;

      back_ = (*back_ == 0) ? array_capacity_ - 1 : *back_ - 1;

      --size_;

      return array_[old_back_];
    }

    //--------------------------------------------------------------------------
    /// \brief Is the deque empty?
    /// \details O(1)
    //--------------------------------------------------------------------------
    virtual bool is_empty() const override;

    //--------------------------------------------------------------------------
    /// \brief Number of items in the deque.
    /// \details O(1)
    //--------------------------------------------------------------------------
    virtual std::size_t size() const override;

  protected:

    bool is_full()
    {
      return size_ == array_capacity_;
    }

    std::size_t get_front_index() const
    {
      return *front_;
    }

    std::size_t get_back_index() const
    {
      return *back_;
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

      if (*front_ <= *back_)
      {
        assert(*front_ == 0 && *back_ == array_capacity_ - 1);

        std::copy(array_, array_ + array_capacity_, tmp_array);
      }
      else
      {
        for (std::size_t i {0}; i < (array_capacity_ - *front_); ++i)
        {
          tmp_array[i] = array_[i + *front_];
        }

        for (std::size_t i {0}; i < *front_; ++i)
        {
          tmp_array[i + (array_capacity_ - *front_)] = array_[i];
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
    std::optional<std::size_t> front_;
    std::optional<std::size_t> back_;
    std::size_t array_capacity_;
    T* array_;
};

template <typename T>
ResizableDeque<T>::ResizableDeque(const std::size_t N):
  size_{0},
  front_{std::nullopt},
  back_{std::nullopt},
  array_capacity_{std::max(N, static_cast<std::size_t>(1))},
  array_{new T[N]}
{}

template <typename T>
ResizableDeque<T>::~ResizableDeque()
{
  delete [] array_;
}

template <typename T>
T ResizableDeque<T>::front() const
{
  if (is_empty())
  {
    throw std::runtime_error("Called front on empty ResizableDeque");
  }

  return array_[*front_];
}

template <typename T>
bool ResizableDeque<T>::is_empty() const
{
  return size_ == 0;
}

template <typename T>
void ResizableDeque<T>::push_back(const T& item)
{
  if (is_full())
  {
    double_capacity();
  }

  if (is_empty())
  {
    front_ = 0;
    back_ = 0;
  }
  else
  {
    back_ = (*back_ + 1) % array_capacity_; 
  }

  array_[*back_] = item;

  ++size_;
}

template <typename T>
T ResizableDeque<T>::pop_front()
{
  if (is_empty())
  {
    throw std::runtime_error("Called pop_front on an empty ResizableDeque");
  }

  if (*front_ == *back_)
  {
    assert(size() == 1);

    back_ = std::nullopt;
    
    const std::size_t old_front_ = *front_;
    front_ = std::nullopt;

    size_ = 0;
    return array_[old_front_];
  }

  const std::size_t old_front_ = *front_;

  front_ = (*front_ + 1) % array_capacity_;

  --size_;

  return array_[old_front_];
}

template <typename T>
std::size_t ResizableDeque<T>::size() const
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