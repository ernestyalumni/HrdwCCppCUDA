#ifndef DATA_STRUCTURES_STATIC_FREE_LIST_H
#define DATA_STRUCTURES_STATIC_FREE_LIST_H

#include "Arrays/DynamicArray.h"
#include "Utilities/KedykUtilities.h"

#include <cassert>
#include <cstddef>

namespace DataStructures
{
namespace Kedyk
{

//------------------------------------------------------------------------------
/// \ref 5.6 Garbage-collecting Free List. pp. 47. Implementing Useful
/// \url https://github.com/dkedyk/ImplementingUsefulAlgorithms/blob/master/Utils/GCFreeList.h
/// Algorithms in C++. Kedyk.
//------------------------------------------------------------------------------
template <typename T>
struct StaticFreeList
{
  // size is current size and max size is largest allocation.
  std::size_t capacity_;
  std::size_t size_;
  std::size_t max_size_;

  struct Item
  {
    T item_;
    union
    {
      // Used when empty cell.
      Item* next_;
      // Used when allocated.
      void* user_data_;
    };
  };

  Item* nodes_;
  Item* returned_;

  StaticFreeList(const std::size_t fixed_size):
    capacity_{fixed_size},
    size_{0},
    max_size_{0},
    // This code line, implementation, may not work:
    // nodes_{new Item[fixed_size]}
    nodes_{Utilities::Kedyk::raw_memory<Item>(fixed_size)},
    returned_{nullptr}
  {}

  bool is_full()
  {
    return size_ == capacity_;
  }

  bool is_empty()
  {
    return size_ == 0;
  }

  Item* allocate()
  {
    // Must handle full blocks externally - check using pointer arithmetic.
    assert(!is_full());
    Item* result {returned_};
    if (result)
    {
      returned_ = returned_->next_;
    }
    else
    {
      result = &nodes_[max_size_++];
    }
    ++size_;
    return result;
  }

  void remove(Item* item)
  {
    // Nodes must come from this list.
    assert(item - nodes_ >= 0 && item - nodes_ < max_size_);
    item->item_.~T();
    item->next_ = returned_;
    returned_ = item;
    --size_;
  }

  virtual ~StaticFreeList()
  {
    if (!is_empty())
    {
      // Mark allocated nodes, unmark returned ones, destruct marked ones.
      DataStructures::Arrays::DynamicArray<bool> to_destruct {
        max_size_,
        true};

      while (returned_)
      {
        // Go through the return list of unmark.
        to_destruct[returned_ - nodes_] = false;
        returned_ = returned_->next_;
      }

      for (std::size_t i {0}; i < max_size_; ++i)
      {
        if (to_destruct[i])
        {
          nodes_[i].item_.~T();
        }
      }
    }
    // This code line, implementation, may not work.
    // delete [] nodes_;
    Utilities::Kedyk::raw_delete(nodes_);
  }
};

} // namespace Kedyk
} // namespace DataStructures

#endif // DATA_STRUCTURES_STATIC_FREE_LIST_H