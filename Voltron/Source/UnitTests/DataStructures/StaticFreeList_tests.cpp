#include "DataStructures/StaticFreeList.h"

#include <boost/test/unit_test.hpp>
#include <string>

BOOST_AUTO_TEST_SUITE(DataStructures)

BOOST_AUTO_TEST_SUITE(Kedyk)

BOOST_AUTO_TEST_SUITE(StaticFreeList_tests)

using namespace DataStructures;

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(Constructs)
{
  Kedyk::StaticFreeList<std::string> static_free_list {42};
  BOOST_TEST(static_free_list.capacity_ == 42);
  BOOST_TEST(static_free_list.size_ == 0);
  BOOST_TEST(static_free_list.max_size_ == 0);
  BOOST_TEST(static_free_list.returned_ == nullptr);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(AllocateAllocates)
{
  Kedyk::StaticFreeList<std::string> static_free_list {42};
  auto result = static_free_list.allocate(); 
  BOOST_TEST(static_free_list.max_size_ == 1);
  BOOST_TEST(static_free_list.size_ == 1);
  result->item_ = "Tribe Called Quest";
  BOOST_TEST(static_free_list.nodes_[0].item_ == "Tribe Called Quest");
  BOOST_TEST(static_free_list.returned_ == nullptr);

  result = static_free_list.allocate(); 
  BOOST_TEST(static_free_list.max_size_ == 2);
  BOOST_TEST(static_free_list.size_ == 2);
  result->item_ = "De La Soul";
  BOOST_TEST(static_free_list.nodes_[1].item_ == "De La Soul");
  BOOST_TEST(static_free_list.returned_ == nullptr);

  result = static_free_list.allocate(); 
  BOOST_TEST(static_free_list.max_size_ == 3);
  BOOST_TEST(static_free_list.size_ == 3);
  result->item_ = "African Soul Child";
  BOOST_TEST(static_free_list.nodes_[2].item_ == "African Soul Child");
  BOOST_TEST(static_free_list.returned_ == nullptr);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(RemoveWorksAfterAllocate)
{
  Kedyk::StaticFreeList<std::string> static_free_list {42};
  auto result = static_free_list.allocate(); 
  static_free_list.remove(result);
  BOOST_TEST(static_free_list.max_size_ == 1);
  BOOST_TEST(static_free_list.size_ == 0);
  BOOST_TEST(result->next_ == nullptr);
  BOOST_TEST(static_free_list.returned_ != nullptr);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(AllocateWorksAfterRemove)
{
  Kedyk::StaticFreeList<std::string> static_free_list {42};
  auto result = static_free_list.allocate(); 
  static_free_list.remove(result);
  result = static_free_list.allocate();
  BOOST_TEST(static_free_list.max_size_ == 1);
  BOOST_TEST(static_free_list.size_ == 1);
  result->item_ = "Tribe Called Quest";
  BOOST_TEST(static_free_list.nodes_[0].item_ == "Tribe Called Quest");
  BOOST_TEST(static_free_list.returned_ == nullptr);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(AllocateToFull)
{
  Kedyk::StaticFreeList<std::string> static_free_list {12};
  BOOST_TEST(static_free_list.is_empty());
  auto result = static_free_list.allocate(); 
  
  /*
  BOOST_TEST(!static_free_list.is_full());
  result = static_free_list.allocate()
  BOOST_TEST(!static_free_list.is_full());
  result = static_free_list.allocate()
  BOOST_TEST(static_free_list.is_full());
  */
}

BOOST_AUTO_TEST_SUITE_END() // StaticFreeList_tests
BOOST_AUTO_TEST_SUITE_END() // Kedyk

BOOST_AUTO_TEST_SUITE_END() // DataStructures