#ifndef DATA_STRUCTURES_QUEUES_MULTIQUEUE_H
#define DATA_STRUCTURES_QUEUES_MULTIQUEUE_H

#include "DynamicQueue.h"

#include <algorithm> // std::min
#include <cstddef>
#include <optional>

namespace DataStructures
{
namespace Queues
{
namespace DWHarder
{

//------------------------------------------------------------------------------
/// \url https://ece.uwaterloo.ca/~dwharder/aads/Lecture_materials/7.01.Priority_queues.pptx
/// https://ece.uwaterloo.ca/~dwharder/aads/Lecture_materials/7.01.Priority_queues.pdf
/// \details Multiple Queues was considered for a Priority Queue, but there were
/// problems compared to heaps.
/// Assume there's a fixed number of priorities, say M.
/// Create an array of M queues.
/// Push a new object onto the queue corresponding to priority.
///
/// Cons:
/// Restricts range of priorities.
/// Memory requirements seems to be O(M*N), not O(M +N) like Harder said in
/// slide 13 of 7.01.Priority_queues.pptx, Abstract Priority Queues.
//------------------------------------------------------------------------------

template <typename T, std::size_t M>
class MultiQueue
{
  public:

    using Queue = DataStructures::Queues::AsHierarchy::DynamicQueue<T>;

    MultiQueue():
      queue_array_{new Queue[M]},
      queue_size_{0},
      current_top_{std::nullopt}
    {}

    MultiQueue(const MultiQueue&) = delete;
    MultiQueue& operator=(const MultiQueue&) = delete;
    MultiQueue(MultiQueue&&) = delete;
    MultiQueue& operator=(MultiQueue&&) = delete;

    virtual ~MultiQueue()
    {
      delete [] queue_array_;
    }

    bool is_empty() const
    {
      return queue_size_ == 0;
    }

    //--------------------------------------------------------------------------
    /// \ref 7.01.Priority_queues.Questions.pdf, 2013 D.W. Harder. ECE 250. 7.1a
    //--------------------------------------------------------------------------
    T top() const
    {
      if (is_empty())
      {
        throw std::runtime_error("Failed to call top on empty MultiQueue");
      }

      return queue_array_[current_top_].front();
    }

    //--------------------------------------------------------------------------
    /// \details Push is O(1) time complexity.
    /// \ref Slide 13, 7.01.Priority_queues.pptx, Abstract Priority Queues.
    //--------------------------------------------------------------------------
    void push(const T object, const std::size_t priority)
    {
      if (priority >= M)
      {
        throw std::runtime_error(
          "Illegal argument for priority input value in MultiQueue");
      }

      queue_array_[priority].push(object);


      current_top_ = current_top_.has_value() ?
        std::min(*current_top_, priority) : priority;
      ++queue_size_;
    }

    //--------------------------------------------------------------------------
    /// \ref 7.01.Priority_queues.Questions.pdf, 2013 D.W. Harder. ECE 250. 7.1a
    //--------------------------------------------------------------------------
    T pop()
    {
      if (is_empty())
      {
        throw std::runtime_error("Failed to pop for empty MultiQueue");
      }

      const T result {queue_array_[current_top_].pop()};
      --queue_size_;

      if (is_empty())
      {
        current_top_ = std::nullopt;
        return result;        
      }

      if (!queue_array_[current_top_].is_empty())
      {
        return result;
      }

      for (std::size_t priority {*current_top_}; priority < M; ++priority)
      {
        if (!queue_array_[priority].is_empty())
        {
          current_top_ = priority;
          return result;
        }
      }

      throw std::runtime_error(
        "Failed to find queue element when queue size is nonzero in MultiQueue");
    }

    std::size_t get_current_top_priority()
    {
      if (!current_top_.has_value())
      {
        throw std::runtime_error(
          "Failed to get value for current top in MultiQueue");
      }

      return *current_top_;
    }

  protected:

    //--------------------------------------------------------------------------
    /// \details Top is O(M) time complexity.
    /// \ref Slide 13, 7.01.Priority_queues.pptx, Abstract Priority Queues.
    //--------------------------------------------------------------------------
    T old_top() const
    {
      for (std::size_t priority {0}; priority < M; ++priority)
      {
        if (!queue_array_[priority].is_empty())
        {
          return queue_array_[priority].front();
        }
      }

      // The priority queue is empty.
      throw std::runtime_error("called old_top on empty MultiQueue.");
    }

    //--------------------------------------------------------------------------
    /// \details Pop is O(M) time complexity.
    /// \ref Slide 13, 7.01.Priority_queues.pptx, Abstract Priority Queues.
    //--------------------------------------------------------------------------
    T old_pop()
    {
      for (std::size_t priority {0}; priority < M; ++priority)
      {
        if (!queue_array_[priority].is_empty())
        {
          --queue_size_;
          return queue_array_[priority].pop();
        }
      }

      throw std::runtime_error("Failed to pop for MultiQueue");
    }

  private:

    Queue* queue_array_;
    std::size_t queue_size_;
    std::optional<std::size_t> current_top_;

}; // class MultiQueue

} // namespace DWHarder
} // namespace Queues
} // namespace DataStructures

#endif // DATA_STRUCTURES_QUEUES_MULTIQUEUE_H
