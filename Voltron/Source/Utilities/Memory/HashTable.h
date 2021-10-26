//------------------------------------------------------------------------------
/// \ref ece250.h, Introductory Project of ECE250, U. Waterloo, D.W. Harder.
//------------------------------------------------------------------------------
#ifndef UTILITIES_MEMORY_ALLOCATION_HASH_TABLE_H
#define UTILITIES_MEMORY_ALLOCATION_HASH_TABLE_H

#include <cstddef>

namespace Utilities
{
namespace Memory
{

int to_int(int* ptr);

struct Allocation
{
  void* address_;
  std::size_t size_;
  bool is_array_;
  bool deleted_;

  Allocation();

  Allocation(void* a, const std::size_t s, const bool i);
};

//------------------------------------------------------------------------------
/// \details All instances of an allocation are stored in this chained hash
/// table.
///
/// This class makes as little chances to the original implementation by D.W.
/// Harder for ECE250, U. Waterloo.
//------------------------------------------------------------------------------
class HashTable
{
  public:

    static std::size_t memory_allocation_storage_;

    //--------------------------------------------------------------------------
    /// \details Initialize all of the addresses to 0.
    //--------------------------------------------------------------------------
    HashTable(const std::size_t as);

    virtual ~HashTable();

    void reserve(const std::size_t N);

    std::size_t memory_allocated() const
    {
      return total_memory_allocated_ - total_memory_deleted_;
    }

    void memory_store() const
    {
      HashTable::memory_allocation_storage_ =
        total_memory_allocated_ - total_memory_deleted_;
    }

    void memory_change(const std::size_t n) const;

    void insert(void* ptr, const std::size_t size, const bool is_array);

    std::size_t remove(void* ptr, bool is_array);

    void summary();

    void details();

    // Start recording memory allocations.

    void start_recording()
    {
      record_ = true;
    }

    // Stop recording memory allocations.

    void stop_recording()
    {
      record_ = false;
    }

    bool is_recording()
    {
      return record_;
    }

    // Accessors

    std::size_t array_size() const
    {
      return array_size_;
    }

  protected:

    int hash_function(void* ptr, const int total_values);

  private:

    std::size_t array_size_;
    Allocation* allocated_;
    std::size_t total_memory_allocated_;
    std::size_t total_memory_deleted_;
    bool record_;
};

} // namespace Memory
} // namespace Utilities

#endif // UTILITIES_MEMORY_ALLOCATION_HASH_TABLE_H
