#ifndef DATA_STRUCTURES_HASH_TABLES_HASH_TABLE_H
#define DATA_STRUCTURES_HASH_TABLES_HASH_TABLE_H

#include <cstddef> // std::size_t

namespace DataStructures
{
namespace HashTables
{

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