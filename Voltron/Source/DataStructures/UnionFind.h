//------------------------------------------------------------------------------
/// \author Ernest Yeung
//-----------------------------------------------------------------------------
#ifndef DATA_STRUCTURES_UNION_JOIN_H
#define DATA_STRUCTURES_UNION_JOIN_H

#include <vector>

namespace DataStructures
{

struct UnionFind
{

  std::vector<int> parent_;
  std::vector<int> rank_;

  UnionFind(const int n);

  int find(int x);

  bool unite(int x, int y);

  bool is_connected(int x, int y);
};

} // namespace DataStructures

#endif // DATA_STRUCTURES_UNION_JOIN_H