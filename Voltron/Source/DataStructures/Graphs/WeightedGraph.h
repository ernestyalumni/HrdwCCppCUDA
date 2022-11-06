#ifndef DATA_STRUCTURES_GRAPHS_WEIGHTED_GRAPH_H
#define DATA_STRUCTURES_GRAPHS_WEIGHTED_GRAPH_H

#include <algorithm> // std::fill
#include <cstddef>
#include <cstring>
#include <limits>
#include <type_traits> // std::is_floating_point

namespace DataStructures
{
namespace Graphs
{

//------------------------------------------------------------------------------
/// \url https://ece.uwaterloo.ca/~dwharder/aads/Projects/5/Dijkstra/src/Weighted_graph.h
/// \ref 11.01. Graph_theory.pptx, Graph theory and the Graph ADT.
//------------------------------------------------------------------------------
template <typename EDGE_VALUE_T>
class InefficientWeightedGraph
{
  public:

    //--------------------------------------------------------------------------
    /// \ref Slide 24, 11.02.Graph data structures, "Default Values."
    /// For vertices not connected, use as default value 0, negative number,
    /// e.g. -1, or positive infinity +oo.
    /// Positive infinity +oo is most logical in that it makes sense 2 vertices
    /// which aren't connected have an infinite distance between them.
    ///
    /// As defined in IEEE 754 standard, representation of double-precision
    /// floating-point infinity 8 bytes:
    /// 0x7F F0 00 00 00 00 00 00;
    /// negative infinity stored as 0x FF F0 00 00 00 00 00 00.
    //--------------------------------------------------------------------------
    inline static constexpr double infinity_ {
      std::numeric_limits<double>::infinity()};

    InefficientWeightedGraph(const std::size_t n):
      n_{n},
      adjacency_matrix_{new EDGE_VALUE_T*[n]}
    {
      for (std::size_t i {0}; i < n; ++i)
      {
        adjacency_matrix_[i] = new EDGE_VALUE_T[n];
      }

      //------------------------------------------------------------------------
      /// \url https://stackoverflow.com/questions/1373369/which-is-faster-preferred-memset-or-for-loop-to-zero-out-an-array-of-doubles#:~:text=memset%20can%20be%20faster%20since,simply%20does%20a%20loop%20internally.
      /// \details memset() is faster since it's written in assembly, std::fill
      /// is a template function which loops internally, but for type safety and
      /// readable code, std::fill() is the C++ way of doing things.
      /// memset() needs you to pass number of bytes, not number of elements,
      /// because it's an old C funciton.
      //------------------------------------------------------------------------

      EDGE_VALUE_T default_value {get_default_value()};
      for (std::size_t i {0}; i < n_; ++i)
      {
        std::fill(
          adjacency_matrix_[i],
          adjacency_matrix_[i] + n_,
          default_value);
      }
    }

    // Copy ctor.
    InefficientWeightedGraph(const InefficientWeightedGraph& other):
      n_{other.n_},
      adjacency_matrix_{new EDGE_VALUE_T*[other.n_]}
    {
      create_arrays();

      for (std::size_t i {0}; i < n_; ++i)
      {
        for (std::size_t j {0}; j < n_; ++j)
        {
          adjacency_matrix_[i][j] = other.adjacency_matrix_[i][j];
        }
      }
    }

    // Copy assignment.
    InefficientWeightedGraph& operator=(const InefficientWeightedGraph& other)
    {
      delete_arrays();
      delete[] adjacency_matrix_;
      n_ = other.n_;
      adjacency_matrix_ = new EDGE_VALUE_T*[other.n_];
      create_arrays();

      for (std::size_t i {0}; i < n_; ++i)
      {
        for (std::size_t j {0}; j < n_; ++j)
        {
          adjacency_matrix_[i][j] = other.adjacency_matrix_[i][j];
        }
      }

      return *this;
    }

    virtual ~InefficientWeightedGraph()
    {
      delete_arrays();
      delete[] adjacency_matrix_;
    }

    EDGE_VALUE_T get_default_value()
    {
      /*
      if (std::is_floating_point_v<EDGE_VALUE_T> ||
        std::is_integral_V<EDGE_VALUE_T>)
      {
        return static_cast<EDGE_VALUE_T>(infinity_);
      }
      */
      if (std::is_same_v<EDGE_VALUE_T, bool>)
      {
        return false;
      }
      
      return static_cast<EDGE_VALUE_T>(infinity_);
    }

  private:

    void create_arrays()
    {
      for (std::size_t i {0}; i < n_; ++i)
      {
        adjacency_matrix_[i] = new EDGE_VALUE_T[n_];
      }
    }

    void delete_arrays()
    {
      for (std::size_t i {0}; i < n_; ++i)
      {
        delete[] adjacency_matrix_[i];
      }
    }

    std::size_t n_;
    EDGE_VALUE_T** adjacency_matrix_;
};

//------------------------------------------------------------------------------
/// \ref 11.02. Graph_data_structures.pptx, Adjacency Matrix Improvement.
//------------------------------------------------------------------------------
template <typename EDGE_VALUE_T>
class WeightedGraph
{
  public:

    //--------------------------------------------------------------------------
    /// \ref Slide 24, 11.02.Graph data structures, "Default Values."
    /// For vertices not connected, use as default value 0, negative number,
    /// e.g. -1, or positive infinity +oo.
    /// Positive infinity +oo is most logical in that it makes sense 2 vertices
    /// which aren't connected have an infinite distance between them.
    ///
    /// As defined in IEEE 754 standard, representation of double-precision
    /// floating-point infinity 8 bytes:
    /// 0x7F F0 00 00 00 00 00 00;
    /// negative infinity stored as 0x FF F0 00 00 00 00 00 00.
    //--------------------------------------------------------------------------
    inline static constexpr double infinity_ {
      std::numeric_limits<double>::infinity()};

    WeightedGraph(const std::size_t n):
      n_{n},
      // Allocate an array of n pointers to EDGE_VALUE_T.
      adjacency_matrix_{new EDGE_VALUE_T*[n]},
      // Allocate an array of n^2 EDGE_VALUE_T.
      adjacency_matrix_values_{new EDGE_VALUE_T[n * n]}
    {
      for (std::size_t i {0}; i < n; ++i)
      {
        // Allocate the addresses.
        adjacency_matrix_[i] = &(adjacency_matrix_values_[n * i]);
      }

      //------------------------------------------------------------------------
      /// \url https://stackoverflow.com/questions/1373369/which-is-faster-preferred-memset-or-for-loop-to-zero-out-an-array-of-doubles#:~:text=memset%20can%20be%20faster%20since,simply%20does%20a%20loop%20internally.
      /// \details memset() is faster since it's written in assembly, std::fill
      /// is a template function which loops internally, but for type safety and
      /// readable code, std::fill() is the C++ way of doing things.
      /// memset() needs you to pass number of bytes, not number of elements,
      /// because it's an old C funciton.
      //------------------------------------------------------------------------
      std::fill(
        adjacency_matrix_values_,
        adjacency_matrix_values_ + n_ * n_,
        get_default_value());
    }

    // Copy ctor.
    WeightedGraph(const WeightedGraph& other):
      n_{other.n_},
      adjacency_matrix_{new EDGE_VALUE_T*[other.n_]},
      adjacency_matrix_values_{new EDGE_VALUE_T[n_ * n_]}
    {
      for (std::size_t i {0}; i < n_; ++i)
      {
        adjacency_matrix_[i] = &(adjacency_matrix_values_[n_ * i]);
      }

      std::copy(
        other.adjacency_matrix_values_,
        other.adjacency_matrix_values_ + other.n_,
        adjacency_matrix_values_);
    }

    // Copy assignment.
    WeightedGraph& operator=(const WeightedGraph& other)
    {
      delete[] adjacency_matrix_[0];
      delete[] adjacency_matrix_;
      n_ = other.n_;
      adjacency_matrix_ = new EDGE_VALUE_T*[other.n_];
      adjacency_matrix_values_ = new EDGE_VALUE_T[n_ * n_];

      std::copy(
        other.adjacency_matrix_values_,
        other.adjacency_matrix_values_ + other.n_,
        adjacency_matrix_values_);

      return *this;
    }

    virtual ~WeightedGraph()
    {
      delete[] adjacency_matrix_[0];
      delete[] adjacency_matrix_;
    }

    std::size_t number_of_vertices() const
    {
      return n_;
    }

    std::size_t number_of_edges() const
    {
      std::size_t count {0};

      for (std::size_t i {0}; i < n_ * n_; ++i)
      {
        if (adjacency_matrix_values_[i] != get_default_value())
        {
          ++count;
        }
      }
      return count;
    }

    static EDGE_VALUE_T get_default_value()
    {
      if (std::is_same_v<EDGE_VALUE_T, bool>)
      {
        return false;
      }
      
      return static_cast<EDGE_VALUE_T>(infinity_);
    }

    void add_edge(
      const std::size_t from,
      const std::size_t to,
      const EDGE_VALUE_T value =
        std::is_same_v<EDGE_VALUE_T, bool> ? true : EDGE_VALUE_T{})
    {
      *(adjacency_matrix_[from] + to) = value;
    }

    void delete_edge(const std::size_t from, const std::size_t to)
    {
      *(adjacency_matrix_[from] + to) = get_default_value();
    }

    EDGE_VALUE_T get_weight(const std::size_t from, const std::size_t to)
    {
      return *(adjacency_matrix_[from] + to);
    }

    bool is_edge(const std::size_t from, const std::size_t to)
    {
      return *(adjacency_matrix_[from] + to) != get_default_value();
    }

  private:

    std::size_t n_;
    // Allocate an array of V pointers to EDGE_VALUE_T.
    EDGE_VALUE_T** adjacency_matrix_;
    // Allocate an array of V^2 EDGE_VALUE_T.
    EDGE_VALUE_T* adjacency_matrix_values_;
};

} // namespace Graphs
} // namespace DataStructures

#endif // DATA_STRUCTURES_GRAPHS_WEIGHTED_GRAPH_H