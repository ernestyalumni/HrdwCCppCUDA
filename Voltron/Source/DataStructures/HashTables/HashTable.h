#ifndef DATA_STRUCTURES_HASH_TABLES_HASH_TABLE_H
#define DATA_STRUCTURES_HASH_TABLES_HASH_TABLE_H

#include "DataStructures/Arrays/FixedSizeArrays.h"
#include "DataStructures/LinkedLists/DoublyLinkedList.h"

#include <cstddef>

namespace DataStructures
{
namespace HashTables
{

namespace CLRS
{

template <typename T, typename HashFunctionT>
class HashTableWithDoublyLinkedList
{
  public:

    using Chain = DataStructures::LinkedLists::DoublyLinkedList<T>;

    HashTableWithDoublyLinkedList() = delete;

    HashTableWithDoublyLinkedList(
      const std::size_t M,
      HashFunctionT hash_function
      ):
      slots_(M),
      hash_function_{hash_function},
      M_{M}
    {}

    std::size_t get_M() const
    {
      return M_;
    }

  private:

    DataStructures::Arrays::DynamicFixedSizeArray<Chain> slots_;
    HashFunctionT hash_function_;
    std::size_t M_;
};

} // namespace CLRS

template <typename T>
class HashTable
{
  public:

    void insert(const T& value) = 0;

    bool erase(const T& value) = 0;

    void clear() = 0;
};

} // namespace HashTables
} // namespace DataStructures

#endif // DATA_STRUCTURES_HASH_TABLES_HASH_TABLE_H