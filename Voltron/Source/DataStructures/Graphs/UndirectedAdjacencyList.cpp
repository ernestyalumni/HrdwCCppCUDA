#include "UndirectedAdjacencyList.h"

#include <cstddef> // std::size_t
#include <unordered_set>

using std::size_t;
using std::unordered_set;

namespace DataStructures
{
namespace Graphs
{

void UndirectedAdjacencyList::add_edge(const size_t u, size_t v)
{
  // https://en.cppreference.com/w/cpp/container/unordered_set/insert
  // Time Complexity; Average case: O(1), worst O(size())
  adjacency_list_.at(u).insert(v);
  adjacency_list_.at(v).insert(u);
}

size_t UndirectedAdjacencyList::delete_edge(const size_t u, const size_t v)
{
  size_t removed {adjacency_list_.at(u).erase(v)};

  return removed + adjacency_list_.at(v).erase(u);
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
