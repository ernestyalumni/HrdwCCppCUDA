#include "UndirectedAdjacencyList.h"

#include <cstddef> // std::size_t
#include <unordered_set>

using std::size_t;
using std::unordered_set;

namespace DataStructures
{
namespace Graphs
{

auto UndirectedAdjacencyList::add_edge(const size_t u, size_t v)
{
  const unordered_set<size_t> new_edge {u, v};

  // https://en.cppreference.com/w/cpp/container/unordered_set/insert
  // Time Complexity; Average case: O(1), worst O(size())
  return adjacency_list_.insert(new_edge);
}

size_t UndirectedAdjacencyList::delete_edge(const size_t u, const size_t v)
{
  const unordered_set<size_t> edge_to_remove {u, v};

  return adjacency_list_.erase(edge_to_remove);
}

size_t UndirectedAdjacencyList::get_number_of_edges() const
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
