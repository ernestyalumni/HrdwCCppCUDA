#ifndef DATA_STRUCTURES_GRAPHS_ADJACENCY_ARRAY_GRAPH_H
#define DATA_STRUCTURES_GRAPHS_ADJACENCY_ARRAY_GRAPH_H

#include "DataStructures/Arrays/DynamicArray.h"

#include <cstddef> // std::size_t

namespace DataStructures
{
namespace Graphs
{
namespace Kedyk
{

//------------------------------------------------------------------------------
/// \ref Ch. 11 Graph Algorithms, pp. 146-147 of Implementing Useful Algorithms
/// in C++, Kedyk.
//------------------------------------------------------------------------------
template <typename EDGE_DATA_T>
class AdjacencyArrayGraph
{
  public:

    template <typename T>
    using DynamicArray =
      typename DataStructures::Arrays::PrimitiveDynamicArray<T>;

    struct Edge
    {
      std::size_t to_;
      EDGE_DATA_T edge_data_;
      Edge(const std::size_t to_input, const EDGE_DATA_T& edge_data_input):
        to_{to_input},
        edge_data_{edge_data_input}
      {}

      //------------------------------------------------------------------------
      /// \ref https://stackoverflow.com/questions/3575458/does-new-call-default-constructor-in-c
      /// \details new[] calls default constructor for each element except for
      /// build-in types; therefore, it needs a default constructor.
      /// \ref https://isocpp.org/wiki/faq/ctors#arrays-call-default-ctor
      /// \details arrays call default ctor.
      //------------------------------------------------------------------------
      Edge():
        to_{0},
        edge_data_{}
      {}
    };

    AdjacencyArrayGraph(const std::size_t initial_size = 0):
      vertices_{initial_size}
    {}

    // Initialize a graph of n vertices.
    void initialize(const std::size_t n)
    {
      for (std::size_t i {0}; i < n; ++i)
      {
        add_vertex();
      }
    }

    std::size_t number_of_vertices() const
    {
      return vertices_.size();
    }

    std::size_t number_of_edges(const std::size_t v) const
    {
      return vertices_[v].size();
    }

    void add_vertex()
    {
      vertices_.append(DynamicArray<Edge>{});
    }

    void add_edge(
      const std::size_t from,
      const std::size_t to,
      const EDGE_DATA_T& edge_data = EDGE_DATA_T{})
    {
      assert(to < vertices_.size());
      vertices_[from].append(Edge(to, edge_data));
    }

    void add_undirected_edge(
      const std::size_t from,
      const std::size_t to,
      const EDGE_DATA_T& edge_data = EDGE_DATA_T{})
    {
      add_edge(from, to, edge_data);
      add_edge(to, from, edge_data);
    }

    bool is_edge(const std::size_t from, const std::size_t to)
    {
      return (from < vertices_.size() && to < vertices_[from].size());
    }

    class AdjacencyIterator
    {
      public:

        AdjacencyIterator(
          const AdjacencyArrayGraph& g,
          const std::size_t v,
          const std::size_t j_input
          ):
          edges_{&g.vertices[v]},
          j_{j_input}
        {}

        AdjacencyIterator& operator++()
        {
          assert(j_ < edges_.size());
          ++j_;
          return *this;
        }

        std::size_t to()
        {
          return (*edges_)[j_].to_;
        }

        const EDGE_DATA_T* data()
        {
          return (*edges_)[j_].edge_data_;
        }

        bool operator!=(const AdjacencyIterator& rhs)
        {
          return j_ != rhs.j_;
        }

      private:

        const DynamicArray<Edge>* edges_;
        std::size_t j_;
    };

  AdjacencyIterator begin(const std::size_t v) const
  {
    return AdjacencyIterator(*this, v, 0);
  }

  AdjacencyIterator end(const std::size_t v) const
  {
    return AdjacencyIterator(*this, v, number_of_edges(v));
  }

  private:

    DynamicArray<DynamicArray<Edge>> vertices_;
};

} // namespace Kedyk

} // namespace Graphs
} // namespace DataStructures

#endif // DATA_STRUCTURES_GRAPHS_ADJACENCY_ARRAY_GRAPH_H