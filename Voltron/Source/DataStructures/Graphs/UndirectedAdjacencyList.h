#ifndef DATA_STRUCTURES_GRAPHS_UNDIRECTED_ADJACENCY_LIST_H
#define DATA_STRUCTURES_GRAPHS_UNDIRECTED_ADJACENCY_LIST_H

#include <unordered_set>
#include <vector>

#include <cstddef>

namespace DataStructures
{
namespace Graphs
{

class UndirectedAdjacencyList
{
  public:

    UndirectedAdjacencyList(const std::size_t V = 0):
      adjacency_list_{V}
    {}

    virtual ~UndirectedAdjacencyList() = default;

    auto add_edge(const std::size_t u, const std::size_t v);

    //--------------------------------------------------------------------------
    /// \return number of elements removed (0 or 1)
    //--------------------------------------------------------------------------
    std::size_t delete_edge(const std::size_t u, const std::size_t v);

    inline std::size_t get_number_of_edges() const
    {
      return adjacency_list_.size();
    }

    inline bool is_edge(const std::size_t u, const std::size_t v) const
    {
      return adjacency_list_.contains(std::unordered_set<std::size_t>{u, v});
    }

  private:

    // https://stackoverflow.com/questions/27757164/creating-unordered-set-of-unordered-set
    std::unordered_set<std::unordered_set<std::size_t>> adjacency_list_;
};

} // namespace Graphs
} // namespace DataStructures

#endif // DATA_STRUCTURES_GRAPHS_UNDIRECTED_ADJACENCY_LIST_H