#ifndef DATA_STRUCTURES_GRAPHS_GRAPH_H
#define DATA_STRUCTURES_GRAPHS_GRAPH_H

#include <cstddef> // std::size_t

namespace DataStructures
{
namespace Graphs
{
namespace Shaffer
{

//------------------------------------------------------------------------------
/// \brief Graph abstract class. This ADT assumes that the number of vertices is
/// fixed when the graph is created.
/// \ref Sec. 11.2. Graph Implementations. pp. 387. C++ 3rd Ed. Shaffer.
/// \details A graph ADT (Abstract Data Type). This ADT assumes that number of
/// vertices is fixed when the graph is created, but that edges can be added and
/// removed. It also supports a mark array to aid graph traversal algorithms.
//------------------------------------------------------------------------------
class Graph
{
  public:

    // Default constructor.
    Graph() = default;

    // No copy.
    Graph(const Graph&) = delete;

    // No assignment.
    Graph& operator=(const Graph&) = delete;

    // Base destructor.
    virtual ~Graph() = default;

    // Initialize a graph of n vertices.
    virtual void initialize(const std::size_t n) = 0;

    // Return: the number of vertices asnd edges.
    virtual std::size_t n() = 0;
    virtual std::size_t e() = 0;

    // Return v's first neighbor.
    virtual std::size_t first_neighbor(const std::size_t v) = 0;

    // Return v's next neighbor.
    virtual std::size_t next(const std::size_t v, std::size_t w) = 0;

    //--------------------------------------------------------------------------
    /// \brief Set the weight for an edge.
    /// \param i, j: the vertices
    /// \param weight: edge weight.
    //--------------------------------------------------------------------------
    virtual void set_edge_weight(
      const std::size_t v1,
      const std::size_t v2,
      const int weight) = 0;

    //--------------------------------------------------------------------------
    /// \brief Delete an edge.
    /// \param i, j: the vertices
    //--------------------------------------------------------------------------
    virtual void delete_edge(const std::size_t v1, const std::size_t v2) = 0;

    //--------------------------------------------------------------------------
    /// \brief Determine if an edge is in the graph.
    /// \param i, j: the vertices.
    /// \return True if edge i, j has non-zero weight.
    //--------------------------------------------------------------------------
    virtual bool is_edge(const std::size_t i, const std::size_t j) = 0;

    //--------------------------------------------------------------------------
    /// \brief Return an edge's weight.
    /// \param i,j : the vertices.
    /// \return The weight of edge i, j, or zero.
    //--------------------------------------------------------------------------
    virtual int get_weight(const std::size_t i, const std::size_t j) = 0;

    //--------------------------------------------------------------------------
    /// \brief Get and Set the mark value for a vertex.
    /// \param v: the vertex.
    /// \param value: the value to set
    //--------------------------------------------------------------------------
    virtual int get_mark(const std::size_t v) = 0;
    virtual void set_mark(const std::size_t v, const int value) = 0;
};

} // namespace Shaffer

} // namespace Graphs
} // namespace DataStructures

#endif // DATA_STRUCTURES_GRAPHS_GRAPH_H