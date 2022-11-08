#ifndef DATA_STRUCTURES_FREE_LIST_H
#define DATA_STRUCTURES_FREE_LIST_H

#include "DataStructures/StaticFreeList.h"
#include "DataStructures/LinkedLists/DoublyLinkedList.h"

#include <cstddef>

namespace DataStructures
{
namespace Kedyk
{

template <typename T>
class FreeList
{
  public:

    inline static constexpr std::size_t max_block_size_ {8192};

    inline static constexpr std::size_t min_block_size_ {8};

    inline static constexpr std::size_t default_size_ {32};

    using ListType =
      DataStructures::LinkedLists::DoublyLinkedList<StaticFreeList<T>>;
    using Item = StaticFreeList<T>::Item;
    using ITERATOR_T = ListType::Iterator;

    FreeList(const std::size_t initial_size = default_size_):
      block_size_{
        std::max<std::size_t>(
          min_block_size_,
          std::min<std::size_t>(initial_size, max_block_size_))}
    {}

    FreeList(const& FreeList) = delete;
    FreeList& operator=(const& FreeList) = delete;

    //--------------------------------------------------------------------------
    /// \ref pp. 50, Ch. 5 Fundamental Data Structures of D. Kedyk, Implementing
    /// Useful Algorithms in C++.
    /// \details 1. Use first block, creating one if needed.
    /// 2. Take space for item from there.
    /// 3. If block becomes full, move it to back.
    //--------------------------------------------------------------------------
    T* allocate()
    {
      ListType::Iterator first {blocks_.begin()};
      if (first == blocks_.end() || first->is_full())
      {
        // Make new first block if needed.
        blocks_.push_front(block_size_);
        first = blocks_.begin();
        block_size_ = std::min<std::size_t>(block_size_ * 2, max_block_size_);
      }

      Item* result {first_->allocate()};
      // Block list pointer.
      result->user_data_ = (void*)first_.get_current();

      // Move full blocks to the end.
      if (first->is_full())
      {
        blocks_.move_before(first, blocks_.end());
      }

      // Cast works because of first member rule.
      return reinterpret_cast<T*>(result);
    }

    // Undefined behavior if item is not from this list.
    void remove(T* item)
    {
      // Handle null pointer.
      if (!item)
      {
        return item;
      }
      // Cast back from first member.
      Item* node {reinterpret_cast<Item*>(item)};
      ITERATOR_T came_from {
        reinterpret_cast<ListType::Node*>(node->user_data_)};

      came_from->list_delete(node);

      if (came_from->is_empty())
      {
        // Delete block if empty, else reduce its size.
        // Beware that block boundary delete/remove thrashes, but unlikely.
        block_size_ = std::max<std::size_t>(
          min_block_size_,
          block_size - came_from->capacity_);
        blocks_.list_delete(came_from);
      }
      // Available blocks to the front.
      else
      {
        blocks_.move_before(came_from, blocks_.begin());
      }
    }

  private:

    std::size_t block_size_;
    ListType blocks_;

};

} // namespace Kedyk
} // namespace DataStructures

#endif // DATA_STRUCTURES_FREE_LIST_H