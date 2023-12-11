#include "AdjacencyList.h"

#include <cstddef> // std::size_t
#include <unordered_set>

using std::size_t;
using std::unordered_set;

namespace DataStructures
{
namespace Graphs
{

size_t AdjacencyList::add_vertex()
{
  adjacency_list_.emplace_back(unordered_set<size_t>{});

  return adjacency_list_.size() - 1;  
}

bool AdjacencyList::add_edge(const size_t u, size_t v)
{
  // Time complexity: constant on average, worse case O(N).
  if (adjacency_list_.at(u).contains(v))
  {
    return false;
  }

  // https://en.cppreference.com/w/cpp/container/unordered_set/emplace
  // Time complexity: constant on average, worse case O(N).
  adjacency_list_[u].emplace(v);

  return true;
}

bool AdjacencyList::delete_edge(const size_t u, const size_t v)
{
  // Time complexity: constant on average, worse case O(N).
  if (!adjacency_list_.at(u).contains(v))
  {
    return false;
  }

  adjacency_list_.at(u).erase(v);

  return true;
}

size_t AdjacencyList::get_number_of_edges() const
{
  size_t count {0};

  for (const auto& connected_vertices : adjacency_list_)
  {
    count += connected_vertices.size();
  }

  return count;
}

} // namespace Graphs
} // namespace DataStructures
