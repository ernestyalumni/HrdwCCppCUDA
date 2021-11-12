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
#ifndef DATA_STRUCTURES_QUEUES_QUEUES_H
#define DATA_STRUCTURES_QUEUES_QUEUES_H

#include "DataStructures/Arrays.h"

#include <cstddef> // std::size_t
#include <memory>

namespace DataStructures
{
namespace Queues
{

//-----------------------------------------------------------------------------
/// \url https://leetcode.com/explore/learn/card/queue-stack/228/first-in-first-out-data-structure/1337/discuss/153529/Straightforward-Implementation-in-C++-20ms
//-----------------------------------------------------------------------------
class QueueWithVector
{
  public:

    QueueWithVector(int k):
      data_{},
      head_{0},
      tail_{0},
      reset_{true}
    {
      data_.resize(k);
    }

    // Insert an element into the circular queue. Return true if the operation
    // is successful.
    bool enqueue(int value)
    {
      if (is_full())
      {
        return false;
      }

      // Update the reset value when first enqueue happens.
      if (head_ == tail_ && reset_)
      {
        reset_ = false;
      }
      data_[tail_] = value;
      tail_ = (tail_ + 1) % data_.size();
      return true;
    }

    // Delete an element from the circular queue. Return true if the operation
    // is successful.
    bool dequeue()
    {
      if (is_empty())
      {
        return false;
      }

      head_ = (head_ + 1) % data_.size();

      // Update the reset value when last dequeue happens.
      if (head_ == tail_ && !reset_)
      {
        reset_ = true;
      }

      return true;
    }

    // Get the front item from the queue.
    int front() const
    {
      if (is_empty())
      {
        return -1;
      }
      return data_[head_];
    }

    // Get the last item from the queue.
    int rear() const
    {
      if (is_empty())
      {
        return -1;
      }
      return data_[(tail_ + data_.size() - 1) % data_.size()];
    }

    // Checks whether the circular queue is empty or not.
    bool is_empty() const
    {
      if (tail_ == head_ && reset_)
      {
        return true;
      }
      return false;
    }

    // Checks whether the circular queue is full or not.
    bool is_full()
    {
      if (tail_ == head_ && !reset_)
      {
        return true;
      }
      return false;
    }

  private:

    std::vector<int> data_;
    int head_;
    int tail_;
    // reset is the mark when the queue is empty, to differentiate from queue
    // is full, because in both conditions (tail_ == head_) holds.
    // reset_ is true when it's not full yet.
    bool reset_;
};

template <typename T>
class QueueAsResizeable
{
  public:

    QueueAsResizeable():
      data_{},
      p_start_{0}
      //head_{0},
      //tail_{0},
      //reset_{true}
    {}

    // Insert an element into the queue. Return true if operation is successful.
    bool enqueue(T x)
    {
      //return true;
      /*
      if (head_ < 0)
      {
        ++head_;
      }

      ++tail_;
      */
      data_.append(x);
      return true;

      /*
      if (!is_full())
      {
        ++tail_;

        data_.append(x);
      }
      else
      {
        tail_ = 0;
        data_.set(0, x); 
      }
      */

      /*
      if (is_full())
      {
        return false;
      }
      */
      /*
      if (head_ == tail_ && reset_)
      {
        reset_ == false;
      }

      data_[]
      */
    }

    // Delete an element from the queue. Return true if operation is successful.
    bool dequeue()
    {
      if (is_empty())
      {
        return false;
      }

      ++p_start_;
      return true;
      /*
      if (head_ < data_.size() - 1)
      {
        ++head_;
      }
      else
      {
        head_ = 0;
      }
      return true;
      */
    }

    // Get the front item from the queue.
    T front()
    {
      return data_[p_start_];
    }

    // Checks whether the queue is empty or not.
    
    bool is_empty()
    {
      return p_start_ >= data_.size();

      //const bool no_initialization {head_ < 0};      

      //return no_initialization || head_ >= data_.size();
    }
    
    /*
    bool is_full()
    {
      const bool tail_at_end {tail_ == data.capacity() - 1};
      const bool tail_behind_head {tail_ == head_ - 1};

      //return data_.size() == data_.capacity();

      return tail_at_end || tail_behind_head;
    }*/
    
    ssize_t p_start() const
    {
      return p_start_;
    }
    
    /*
    ssize_t head() const
    {
      return head_;
    }

    ssize_t tail() const
    {
      return tail_;
    }*/


    int size() const
    {
      return data_.size();
    }

    int capacity() const
    {
      return data_.capacity();
    }

  private:

    Arrays::ResizingArray<T> data_;
    
    // A "pointer" to indicate the start position.
    ssize_t p_start_;

    //ssize_t head_;
    //ssize_t tail_;
};

//------------------------------------------------------------------------------
/// https://leetcode.com/explore/learn/card/queue-stack/228/first-in-first-out-data-structure/1395/
//------------------------------------------------------------------------------
template <typename T>
class CircularQueue
{
  public:

    explicit CircularQueue(int k):
      data_{},
      head_{-1},
      tail_{-1},
      size_{k}
    {
      data_.resize(size_);
    }

    bool enqueue(T value)
    {
      if (is_full())
      {
        return false;
      }
      if (is_empty())
      {
        head_ = 0;
      }
      tail_ = (tail_ + 1) % size_;
      data_[tail_] = value;
      return true;
    }

    // Delete an element from the circular queue.
    bool dequeue()
    {
      if (is_empty())
      {
        return false;
      }

      if (head_ == tail_)
      {
        head_ = -1;
        tail_ = -1;
        return true;
      }
      head_ = (head_ + 1) % size_;
      return true;
    }

    T front()
    {
      if (is_empty())
      {
        return static_cast<T>(-1);
      }

      return data_[head_];
    }

    T rear()
    {
      if (is_empty())
      {
        return static_cast<T>(-1);
      }
      return data_[tail_];
    }

    // Checks whether the circular queue is empty or not.
    bool is_empty()
    {
      return head_ == -1;
    }

    bool is_full()
    {
      return ((tail_ + 1) % size_) == head_;
    }

  private:

    std::vector<T> data_;
    int head_;
    int tail_;
    int size_;
};

} // namespace Queues
} // namespace DataStructures

#endif // DATA_STRUCTURES_QUEUES_H