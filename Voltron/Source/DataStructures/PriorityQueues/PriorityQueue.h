#ifndef DATA_STRUCTURES_PRIORITY_QUEUES_PRIORITY_QUEUE_H
#define DATA_STRUCTURES_PRIORITY_QUEUES_PRIORITY_QUEUE_H

#include <optional>
#include <vector>

namespace DataStructures
{
namespace PriorityQueues
{

template <typename T>
class PriorityQueue
{
  public:

    bool is_empty() const
    {
      return data_.empty();
    }


    const T& top() const {
      if (data_.empty()) {
          throw std::runtime_error("top() on empty priority queue");
      }
      return data_.front();
  }

  void push(const T& value) {
      data_.push_back(value);
      sift_up(data_.size() - 1);
  }

  void pop() {
      if (data_.empty()) {
          throw std::runtime_error("pop() on empty priority queue");
      }
      std::swap(data_.front(), data_.back());
      data_.pop_back();
      if (!data_.empty()) {
          sift_down(0);
      }
  }

  protected:

    void sift_up(std::size_t i)
    {
      while (i > 0)
      {
        // get_parent_index is monotonically decreasing.
        const std::size_t p {get_parent_index<std::size_t>(i)};
        if (data_[p] < data_[i])
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

        if (l < N && data_[largest] < data_[l]) {
            largest = l;
        }
        if (r < N && data_[largest] < data_[r]) {
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
    static std::size_t get_left_index(const U i)
    {
      return 2 * i + 1;
    }

    template <typename U>
    static int get_right_index(const U i)
    {
      return 2 * i + 2;
    }
    
  private:

    std::vector<T> data_;
};

#include <vector>
#include <stdexcept>

template <typename T>
class MyPriorityQueue{
public:
    bool empty() const { return data_.empty(); }
    std::size_t size() const { return data_.size(); }

    const T& top() const {
        if (data_.empty()) {
            throw std::runtime_error("top() on empty priority queue");
        }
        return data_.front();
    }

    void push(const T& value) {
        data_.push_back(value);
        sift_up(data_.size() - 1);
    }

    void pop() {
        if (data_.empty()) {
            throw std::runtime_error("pop() on empty priority queue");
        }
        std::swap(data_.front(), data_.back());
        data_.pop_back();
        if (!data_.empty()) {
            sift_down(0);
        }
    }

private:
    std::vector<T> data_;

    static std::size_t parent(std::size_t i) { return (i - 1) / 2; }
    static std::size_t left(std::size_t i)   { return 2 * i + 1; }
    static std::size_t right(std::size_t i)  { return 2 * i + 2; }

    void sift_up(std::size_t i) {
        while (i > 0) {
            std::size_t p = parent(i);
            if (data_[p] < data_[i]) {
                std::swap(data_[p], data_[i]);
                i = p;
            } else {
                break;
            }
        }
    }

    void sift_down(std::size_t i) {
        const std::size_t n = data_.size();
        while (true) {
            std::size_t largest = i;
            std::size_t l = left(i);
            std::size_t r = right(i);

            if (l < n && data_[largest] < data_[l]) {
                largest = l;
            }
            if (r < n && data_[largest] < data_[r]) {
                largest = r;
            }
            if (largest == i) {
                break;
            }
            std::swap(data_[i], data_[largest]);
            i = largest;
        }
    }
};


} // namespace PriorityQueues
} // namespace DataStructures

#endif // DATA_STRUCTURES_PRIORITY_QUEUES_PRIORITY_QUEUE_H