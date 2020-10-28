//------------------------------------------------------------------------------
/// \file HashSet.h
/// \author Ernest Yeung
/// \brief Classes and functions demonstrating Hash Set
/// \details Hash set is 1 of the 2 different kinds of hash tables (other is
/// hash map.)
///
/// 1 of the implementations of a set data structure to store no repeated
/// values.
///
/// \ref https://leetcode.com/explore/learn/card/hash-table/182/practical-applications/1139/
//-----------------------------------------------------------------------------
#ifndef DATA_STRUCTURES_HASH_SET_H
#define DATA_STRUCTURES_HASH_SET_H

#include <algorithm> // std::copy
#include <array>
#include <iterator> // std::begin, std::end;
#include <list>
#include <stdexcept> // std::runtime_error

namespace DataStructures
{

namespace HashTables
{

class HashSet
{
  public:

    // Number of buckets
    static constexpr int M {10000};

    HashSet();

    void add(int key);

    void remove(int key);

    //--------------------------------------------------------------------------
    /// \brief Returns true if this set contains the specified element.
    //--------------------------------------------------------------------------
    bool contains(int key);

    int hash(int key);

  private:

    std::array<std::list<int>, M> chained_bucket_array_;
};

} // namespace HashTables
} // namespace DataStructures

#endif // DATA_STRUCTURES_HASH_SET_H