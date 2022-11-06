#include "DataStructures/Vector.h"

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(DataStructures)
BOOST_AUTO_TEST_SUITE(Kedyk)
BOOST_AUTO_TEST_SUITE(Vector_tests)

using DataStructures::Kedyk::Vector;

template <typename ITEM_T>
class TestVector : public Vector<ITEM_T>
{
  public:

    using Vector<ITEM_T>::Vector;
    using Vector<ITEM_T>::get_capacity;
};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ClassConstantsAreConstant)
{
  BOOST_TEST(Vector<int>::MIN_CAPACITY == 8);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DefaultConstructor)
{
  Vector<int> a {};

  BOOST_TEST(true);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ConstructsWithSizeAndInitialValue)
{
  // Added copy ctor, and copy assignment, and no longer SIGABRT.
  // fatal error: in
  // "DataStructures/Kedyk/Vector_tests/ConstructsWithSizeAndInitialValue":
  // signal: SIGABRT (application abort requested)
  TestVector<int> a (6, 9);
  BOOST_TEST(a.get_capacity() == 8);
  BOOST_TEST(a.get_size() == 6);
  BOOST_TEST(a[0] == 9);
  BOOST_TEST(a[1] == 9);
  for (std::size_t i {2}; i < 6; ++i)
  {
    BOOST_TEST(a[i] == 9);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(AppendAppends)
{
  TestVector<int> a {};
  a.append(1);
  BOOST_TEST(a.get_capacity() == 8);
  BOOST_TEST(a.get_size() == 1);
  BOOST_TEST(a[0] == 1);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(AppendAppendsPastCapacity)
{
  TestVector<int> a {};
  for (std::size_t i {0}; i < 12; ++i)
  {
    a.append(i + 1);
    BOOST_TEST(a.get_size() == i + 1);
  }

  for (std::size_t i {0}; i < 12; ++i)
  {
    BOOST_TEST(a[i] == i + 1);
  }
}

template <typename EDGE_DATA_T>
struct Edge
{
  std::size_t to_;
  EDGE_DATA_T edge_data_;

  Edge(const std::size_t to_input, const EDGE_DATA_T& edge_data_input):
    to_{to_input},
    edge_data_{edge_data_input}
  {}

  Edge():
    to_{0},
    edge_data_{}
  {}
};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(NestedVectorsConstructs)
{
  TestVector<TestVector<Edge<double>>> adjacency_list {};
  BOOST_TEST(adjacency_list.get_capacity() == 8);
  BOOST_TEST(adjacency_list.get_size() == 0);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(NestedVectorsAppends)
{
  TestVector<Edge<double>> vertex_0 {};
  vertex_0.append(Edge{1, 1.0});
  vertex_0.append(Edge{2, 2.0});
  TestVector<Edge<double>> vertex_1 {};
  vertex_1.append(Edge{2, 2.0});
  vertex_1.append(Edge{3, 3.0});
  TestVector<TestVector<Edge<double>>> adjacency_list {};
  adjacency_list.append(vertex_0);
  adjacency_list.append(vertex_1);
  BOOST_TEST(adjacency_list[0][0].to_ == 1);
  BOOST_TEST(adjacency_list[0][0].edge_data_ == 1.0);
  BOOST_TEST(adjacency_list[0][1].to_ == 2);
  BOOST_TEST(adjacency_list[0][1].edge_data_ == 2.0);
  BOOST_TEST(adjacency_list[1][0].to_ == 2);
  BOOST_TEST(adjacency_list[1][0].edge_data_ == 2.0);
  BOOST_TEST(adjacency_list[1][1].to_ == 3);
  BOOST_TEST(adjacency_list[1][1].edge_data_ == 3.0);
}

BOOST_AUTO_TEST_SUITE_END() // Vector_tests
BOOST_AUTO_TEST_SUITE_END() // Kedyk
BOOST_AUTO_TEST_SUITE_END() // DataStructures