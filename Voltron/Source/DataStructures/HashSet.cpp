//------------------------------------------------------------------------------
/// \file HashSet.cpp
/// \author Ernest Yeung
/// \brief Classes and functions demonstrating hash set
/// \ref https://leetcode.com/explore/learn/card/hash-table/182/practical-applications/1139/
//-----------------------------------------------------------------------------
#include "HashSet.h"

#include <algorithm> // std::find;

using std::find;

namespace DataStructures
{

namespace HashTables
{

HashSet::HashSet() = default;

void HashSet::add(int key)
{
  if (!contains(key))
  {
      auto& bucket = chained_bucket_array_.at(hash(key));
      bucket.emplace_back(key);
  }
}

void HashSet::remove(int key)
{
  auto& bucket = chained_bucket_array_.at(hash(key));
  
  auto iter = find(bucket.begin(), bucket.end(), key);
      
  if (iter != bucket.end())
  {
      bucket.erase(iter);
  }
}
    
/** Returns true if this set contains the specified element */
bool HashSet::contains(int key)
{
  auto& bucket = chained_bucket_array_.at(hash(key));
  
  auto iter = find(bucket.begin(), bucket.end(), key);
  
  return (iter != bucket.end());
}

int HashSet::hash(int key)
{
  return key % M;
}



} // namespace HashTables
} // namespace DataStructures
