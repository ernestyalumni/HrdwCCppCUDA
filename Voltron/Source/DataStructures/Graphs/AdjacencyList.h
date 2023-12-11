#ifndef DATA_STRUCTURES_GRAPHS_ADJACENCY_LIST_H
#define DATA_STRUCTURES_GRAPHS_ADJACENCY_LIST_H

#include <unordered_set>
#include <vector>

#include <cstddef>

namespace DataStructures
{
namespace Graphs
{

class AdjacencyList
{
  public:

    AdjacencyList():
      adjacency_list_{}
    {}

    AdjacencyList(const std::size_t number_of_vertices):
      adjacency_list_(number_of_vertices, std::unordered_set<std::size_t>{})
    {}

    virtual ~AdjacencyList() = default;

    std::size_t add_vertex();

    //--------------------------------------------------------------------------
    /// \details O(N) time complexity.
    /// \returns True if we had not found an existing edge from u to v and we
    /// added it.
    //--------------------------------------------------------------------------
    bool add_edge(const std::size_t u, const std::size_t v);

    bool delete_edge(const std::size_t u, const std::size_t v);

    std::size_t get_number_of_edges() const;

    inline bool is_edge(const std::size_t u, const std::size_t v) const
    {
      return adjacency_list_.at(u).contains(v);
    }

    inline std::size_t get_size() const
    {
      return adjacency_list_.size();
    }

  private:

    std::vector<std::unordered_set<std::size_t>> adjacency_list_;
};

} // namespace Graphs
} // namespace DataStructures

#endif // DATA_STRUCTURES_GRAPHS_ADJACENCY_LIST_H