#include "HashTable.h"

#include <cstddef>
#include <iostream>
#include <stdexcept>

using std::size_t;

namespace Utilities
{
namespace Memory
{

int to_int(int* ptr)
{
  int result {*ptr};

  if (result < 0)
  {
    result = result + (1 << (sizeof(int) - 1));
  }

  return result >> 3;
}

Allocation::Allocation():
  address_{nullptr},
  size_{0},
  is_array_{false},
  deleted_{false}
{}

Allocation::Allocation(void* a, const size_t s, const bool i):
  address_{a},
  size_{s},
  is_array_{i},
  deleted_{false}
{}

std::size_t HashTable::memory_allocation_storage_ {0};

HashTable::HashTable(const size_t as):
  array_size_{as},
  total_memory_allocated_{0},
  total_memory_deleted_{0},
  record_{false}
{
  allocated_ = new Allocation[array_size_];
}

HashTable::~HashTable()
{
  delete [] allocated_;
}

void HashTable::reserve(const size_t N)
{
  // N must be a power of 2
  if ((N & ((~N) + 1)) != N)
  {
    throw std::runtime_error("Illegal argument in reserve for N");
  }

  delete [] allocated_;
  array_size_ = N;
  allocated_ = new Allocation[array_size()];
}

void HashTable::memory_change(const size_t n) const
{
  const size_t memory_allocated_diff {
    total_memory_allocated_ -
    total_memory_deleted_ -
    HashTable::memory_allocation_storage_};

  if (memory_allocated_diff != n)
  {
    std::clog << "WARNING: expecting a change in memory allocation of " <<
      n <<
      " bytes, but the change was " <<
      memory_allocated_diff << '\n';
  }
}

int HashTable::hash_function(void* ptr, const int total_values)
{
  return to_int(reinterpret_cast<int *>(&ptr)) & (total_values - 1);
}

void HashTable::insert(void* ptr, const std::size_t size, const bool is_array)
{
  if (!record_)
  {
    return;
  }

  // The hash function is the last log[2]( array_size ) bits
  int hash {hash_function(ptr, static_cast<int>(array_size_))};

  for (std::size_t i {0}; i < array_size_; ++i)
  {
    // It may be possible that we are allocated the same memory location twice
    // (if there are numerous allocations and dealocations of memory). Thus, the
    // second check is necessary, otherwise it may introduce session dependant
    // errors.

    if (allocated_[hash].address_ == 0 || allocated_[hash].address_ == ptr)
    {
      // Store the address, the amount of memory allocated, whether or not new[]
      // was used, and set 'deleted' to false.

      allocated_[hash] = Allocation{ptr, size, is_array};

      // Add the memory allocated to the total memory allocated.
      total_memory_allocated_ += size;

      return;
    }

    hash = (hash + 1) & (static_cast<int>(array_size_) - 1);
  }
}

} // namespace Memory
} // namespace Utilities