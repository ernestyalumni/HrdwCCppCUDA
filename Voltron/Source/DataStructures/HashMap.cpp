//------------------------------------------------------------------------------
/// \file HashMap.cpp
/// \author Ernest Yeung
/// \brief Classes and functions demonstrating Hash Map
/// \details Hash Map is 1 of the 2 different kinds of hash tables (other is
/// hash set.)
//------------------------------------------------------------------------------
#include "HashMap.h"

#include <cstddef> // std::size_t
#include <utility> // std::make_pair;

using std::make_pair;
using std::move;

namespace DataStructures
{

namespace HashTables
{

namespace HashMap
{

//------------------------------------------------------------------------------
/// \class HashMapListVector
//------------------------------------------------------------------------------

HashMapListVector::HashMapListVector():
  buckets_{},
  length_{default_length_}
{
  buckets_.resize(length_);
}

HashMapListVector::HashMapListVector(const size_t bucket_number):
  buckets_{},
  length_{bucket_number}
{
  buckets_.resize(length_);
}

void HashMapListVector::put(int key, int value)
{
  auto& bucket = buckets_.at(hash_function(key));
  for (auto& element : bucket)
  {
    if (element.first == key)
    {
      element.second = value;
      // Done, no need to iterate through rest of the list - 
      return;
    }
  }

  // Not found, put a new entry.
  bucket.emplace_back(make_pair<int, int>(move(key), move(value)));
}

int HashMapListVector::hash_function(int key)
{
  return key % length_;
}

int HashMapListVector::get(int key)
{
  const auto& bucket = buckets_.at(hash_function(key));

  if (bucket.empty())
  {
    return -1;
  }

  for (const auto& element : bucket)
  {
    if (element.first == key)
    {
      return element.second;
    }
  }

  return -1;
}

void HashMapListVector::remove(int key)
{
  auto& bucket = buckets_.at(hash_function(key));

  bucket.remove_if(
    [key](auto element)
    {
      return element.first == key;
    });
}

} // namespace HashMap
} // namespace HashTables
} // namespace DataStructures
