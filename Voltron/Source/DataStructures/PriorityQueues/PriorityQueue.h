#ifndef DATA_STRUCTURES_PRIORITY_QUEUES_PRIORITY_QUEUE_H
#define DATA_STRUCTURES_PRIORITY_QUEUES_PRIORITY_QUEUE_H

#include <functional> // std::less
#include <optional>
#include <utility>
#include <vector>

namespace DataStructures
{
namespace PriorityQueues
{

template <typename T>
class PriorityQueue
{
  public:

    // Default constructor: max-heap using operator<
    PriorityQueue():
      comparator_{std::less<T>{}},
      data_{}
    {}

    // Constructor with custom comparator (lambda, functor, function ptr, etc.)
    // The comparator comp(a, b) should return true if a has lower priority
    // than b
    template <typename Comparator>
    explicit PriorityQueue(Comparator&& cmp):
      comparator_{std::forward<Comparator>(cmp)},
      data_{}
    {}

    // Addition of this constructor makes call overloaded and ambiguous.
    // template <typename Comparator>
    // explicit PriorityQueue(Comparator comparator):
    //   comparator_{std::move(comparator)},
    //   data_{}
    // {}

    // TODO: Disambiguate from custom comparator constructor.
    // Copy constructor.
    // PriorityQueue(const PriorityQueue& other):
    //   comparator_{std::move(other.comparator_)},
    //   data_{other.data_}
    // {}

    //--------------------------------------------------------------------------
    /// O(1) time
    //--------------------------------------------------------------------------
    bool is_empty() const
    {
      return data_.empty();
    }

    //--------------------------------------------------------------------------
    /// O(1) time
    //--------------------------------------------------------------------------
    std::optional<T> top()
    {
      if (data_.empty())
      {
        return std::nullopt;
      }
      return data_.front();
    }

    std::optional<T> top() const
    {
      if (data_.empty())
      {
        return std::nullopt;
      }
      return data_.front();
    }

    //--------------------------------------------------------------------------
    /// O(log(N)) time
    //--------------------------------------------------------------------------
    void push(const T& value)
    {
      data_.push_back(value);
      sift_up(data_.size() - 1);
    }

    //--------------------------------------------------------------------------
    /// O(log(N)) time
    //--------------------------------------------------------------------------
    void pop()
    {
      if (data_.empty())
      {
        return;
      }

      std::swap(data_.front(), data_.back());
      data_.pop_back();
      if (!data_.empty())
      {
        sift_down(0);
      }
    }

    //--------------------------------------------------------------------------
    /// O(N log(N) + k log(N)) time.
    //--------------------------------------------------------------------------
    std::vector<T> get_top_k(const std::size_t k)
    {
      if (k == 0)
      {
        return {};
      }

      std::vector<T> result {};
      result.reserve(k);

      // Create a local copy of the data because we need to use pop.
      PriorityQueue<T> local_copy {comparator_};
      // O(N log(N)) time, for N elements, log(N) insertion.
      for (const T& value : data_)
      {
        local_copy.push(value);
      }

      for (std::size_t i {0}; i < std::min(k, data_.size()); ++i)
      {
        result.push_back(*local_copy.top());
        // Pop top element and sift down all other elements, to maintain the
        // heap property, meaning the top element is the "largest" (or
        // "smallest") element.
        local_copy.pop();
      }
      return result;
    }

  protected:

    void sift_up(std::size_t i)
    {
      while (i > 0)
      {
        // get_parent_index is monotonically decreasing.
        const std::size_t p {get_parent_index<std::size_t>(i)};
        if (comparator_(data_[p], data_[i]))
        {
            std::swap(data_[p], data_[i]);
            i = p;
        } 
        else
        {
          break;
        }
      }
    }

    void sift_down(std::size_t i)
    {
      const std::size_t N {data_.size()};
      while (true) 
      {
        std::size_t largest {i};
        // get_left_index, get_right_index are monotonically increasing.
        std::size_t l {get_left_index<std::size_t>(i)};
        std::size_t r {get_right_index<std::size_t>(i)};

        // We first check if the values in the left or right indices are
        // "smaller", since we want to "sift" or move our value at index i to
        // be "lower" (i.e. higher index) in the "heap" (or data) than all the
        // other values.
        // Now generalized with comp_(a, b) meaning a has lower priority than b
        // (e.g., for max-heap: a < b; for min-heap: a > b)

        if (l < N && comparator_(data_[largest], data_[l])) {
            largest = l;
        }
        if (r < N && comparator_(data_[largest], data_[r])) {
            largest = r;
        }

        if (largest == i) {
            break;
        }
        std::swap(data_[i], data_[largest]);
        i = largest;
      }
    }

    //--------------------------------------------------------------------------
    /// if 0, (0 - 1) / 2 = -0.5 or 0 (round to zero)
    /// if 1, 2, then 0
    /// if 3, 4, then 1
    /// if 5, 6, then 2; if 7, 8, then 3
    /// if 9, 10, then 4; 11, 12, then 5; if 13, 14, then 6; if 15, 16, then 7 
    //--------------------------------------------------------------------------
    template <typename U>
    static U get_parent_index(const U i)
    {
      return (i - 1) / 2;
    }

    template <typename U>
    static U get_left_index(const U i)
    {
      return 2 * i + 1;
    }

    template <typename U>
    static U get_right_index(const U i)
    {
      return 2 * i + 2;
    }
    
  private:

    std::function<bool(const T&, const T&)> comparator_;

    std::vector<T> data_;
};

} // namespace PriorityQueues
} // namespace DataStructures

#endif // DATA_STRUCTURES_PRIORITY_QUEUES_PRIORITY_QUEUE_H