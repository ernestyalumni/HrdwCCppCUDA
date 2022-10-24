#ifndef DATA_STRUCTURES_GRAPHS_ADJACENT_ARRAY_GRAPH_H
#define DATA_STRUCTURES_GRAPHS_ADJACENT_ARRAY_GRAPH_H

#include <cstddef> // std::size_t

namespace DataStructures
{
namespace Graphs
{
namespace Kedyk
{

template <typename EDGE_DATA_T>
class AdjacentArrayGraph
{
  public:

    struct Edge
    {
      int to_;
      EDGE_DATA_T edge_data_;
      Edge(const int to_input, const EDGE_DATA_T& edge_data_input):
        to_{to_input},
        edge_data_{edge_data_input}
      {}
    };

  private:

};

} // namespace Kedyk

} // namespace Graphs
} // namespace DataStructures

#endif // DATA_STRUCTURES_GRAPHS_ADJACENT_ARRAY_GRAPH_H