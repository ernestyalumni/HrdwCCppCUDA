#include "UnionFind.h"

#include <algorithm> // std::swap
#include <numeric> // std::iota (fill the range with sequentially increasing
// values, starting from value)

using std::swap;

namespace DataStructures
{

UnionFind::UnionFind(int n):
  parent_(n),
  rank_(n, 0)
{
  // parent_[i] = i
  iota(parent_.begin(), parent_.end(), 0);
}

int UnionFind::find(int x)
{
  if (parent_[x] != x)
  {
    parent_[x] = find(parent_[x]);
  }

  return parent_[x];
}

bool UnionFind::unite(int x, int y)
{
  x = find(x);
  y = find(y);

  // Already connected - adding edge = cycle
  if (x == y)
  {
    return false;
  }

  if (rank_[x] < rank_[y])
  {
    swap(x, y);
  }

  parent_[y] = x;

  if (rank_[x] == rank_[y])
  {
    ++rank_[x];
  }

  return true;
}

bool UnionFind::is_connected(int x, int y)
{
  return find(x) == find(y);
}

} // namespace DataStructures
