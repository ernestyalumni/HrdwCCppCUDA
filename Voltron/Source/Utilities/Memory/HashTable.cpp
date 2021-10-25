#include "HashTable.h"

#include <cstddef>
#include <iostream>
#include <stdexcept>

using std::size_t;

namespace Utilities
{
namespace Memory
{

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

} // namespace Memory
} // namespace Utilities