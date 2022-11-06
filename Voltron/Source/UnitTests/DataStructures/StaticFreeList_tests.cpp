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
  Kedyk::StaticFreeList<std::string> static_free_list {8};
  BOOST_TEST(static_free_list.capacity_ == 8);
  BOOST_TEST(static_free_list.size_ == 0);
  BOOST_TEST(static_free_list.max_size_ == 0);
  BOOST_TEST(static_free_list.returned_ == nullptr);
  BOOST_TEST(static_free_list.is_empty());
  BOOST_TEST(!static_free_list.is_full());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(AllocateAllocatesToFull)
{
  Kedyk::StaticFreeList<std::string> static_free_list {3};
  auto* allocated_1 = static_free_list.allocate();
  BOOST_TEST(static_free_list.capacity_ == 3);
  BOOST_TEST(static_free_list.size_ == 1);
  BOOST_TEST(static_free_list.max_size_ == 1);
  BOOST_TEST(static_free_list.returned_ == nullptr);
  BOOST_TEST(!static_free_list.is_empty());
  BOOST_TEST(!static_free_list.is_full());
  allocated_1->item_ = "allocated 1";
  BOOST_TEST(static_free_list.nodes_[0].item_ == "allocated 1");

  auto* allocated_2 = static_free_list.allocate();
  BOOST_TEST(static_free_list.capacity_ == 3);
  BOOST_TEST(static_free_list.size_ == 2);
  BOOST_TEST(static_free_list.max_size_ == 2);
  BOOST_TEST(static_free_list.returned_ == nullptr);
  BOOST_TEST(!static_free_list.is_empty());
  BOOST_TEST(!static_free_list.is_full());
  allocated_2->item_ = "allocated 2";
  BOOST_TEST(static_free_list.nodes_[1].item_ == "allocated 2");

  auto* allocated_3 = static_free_list.allocate();
  BOOST_TEST(static_free_list.capacity_ == 3);
  BOOST_TEST(static_free_list.size_ == 3);
  BOOST_TEST(static_free_list.max_size_ == 3);
  BOOST_TEST(static_free_list.returned_ == nullptr);
  BOOST_TEST(!static_free_list.is_empty());
  BOOST_TEST(static_free_list.is_full());
  allocated_3->item_ = "allocated 3";
  BOOST_TEST(static_free_list.nodes_[2].item_ == "allocated 3");
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(RemoveRemovesFromFullFreeList)
{
  Kedyk::StaticFreeList<std::string> static_free_list {3};
  auto* allocated_1 = static_free_list.allocate();
  auto* allocated_2 = static_free_list.allocate();
  auto* allocated_3 = static_free_list.allocate();
  allocated_1->item_ = "allocated 1";
  allocated_2->item_ = "allocated 2";
  allocated_3->item_ = "allocated 3";

  BOOST_TEST(static_cast<int>(allocated_3 - static_free_list.nodes_) == 2);
  static_free_list.remove(allocated_3);
  BOOST_TEST(allocated_3->next_ == nullptr);
  BOOST_TEST(static_free_list.returned_->item_ == "allocated 3");
  BOOST_TEST(!static_free_list.is_empty());
  BOOST_TEST(!static_free_list.is_full());
  BOOST_TEST(static_free_list.size_ == 2);

  BOOST_TEST(static_cast<int>(allocated_2 - static_free_list.nodes_) == 1);
  static_free_list.remove(allocated_2);
  BOOST_TEST(allocated_2->next_->item_ == "allocated 3");
  BOOST_TEST(static_free_list.returned_->item_ == "allocated 2");
  BOOST_TEST(static_free_list.returned_->next_->item_ == "allocated 3");
  BOOST_TEST(!static_free_list.is_empty());
  BOOST_TEST(!static_free_list.is_full());
  BOOST_TEST(static_free_list.size_ == 1);

  BOOST_TEST(static_cast<int>(allocated_1 - static_free_list.nodes_) == 0);
  static_free_list.remove(allocated_1);
  BOOST_TEST(allocated_1->next_->item_ == "allocated 2");
  BOOST_TEST(static_free_list.returned_->item_ == "allocated 1");
  BOOST_TEST(static_free_list.returned_->next_->item_ == "allocated 2");
  BOOST_TEST(static_free_list.is_empty());
  BOOST_TEST(!static_free_list.is_full());
  BOOST_TEST(static_free_list.size_ == 0);
  BOOST_TEST(static_free_list.size_ == 0);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(RemoveRemovesArbitrarilyFromFullFreeList)
{
  Kedyk::StaticFreeList<std::string> static_free_list {4};
  auto* allocated_1 = static_free_list.allocate();
  BOOST_TEST_REQUIRE(static_free_list.returned_ == nullptr);
  auto* allocated_2 = static_free_list.allocate();
  BOOST_TEST_REQUIRE(static_free_list.returned_ == nullptr);
  auto* allocated_3 = static_free_list.allocate();
  BOOST_TEST_REQUIRE(static_free_list.returned_ == nullptr);
  auto* allocated_4 = static_free_list.allocate();
  BOOST_TEST_REQUIRE(static_free_list.returned_ == nullptr);
  allocated_1->item_ = "allocated 1";
  allocated_2->item_ = "allocated 2";
  allocated_3->item_ = "allocated 3";
  allocated_4->item_ = "allocated 4";
  BOOST_TEST(allocated_1->next_ == nullptr);
  BOOST_TEST(allocated_2->next_ == nullptr);
  BOOST_TEST(allocated_3->next_ == nullptr);
  BOOST_TEST(allocated_4->next_ == nullptr);

  static_free_list.remove(allocated_3);
  BOOST_TEST(allocated_3->next_ == nullptr);
  BOOST_TEST(static_free_list.returned_->item_ == "allocated 3");
  BOOST_TEST(!static_free_list.is_empty());
  BOOST_TEST(!static_free_list.is_full());
  BOOST_TEST(static_free_list.size_ == 3);

  static_free_list.remove(allocated_1);
  BOOST_TEST(allocated_1->next_->item_ == "allocated 3");
  BOOST_TEST(static_free_list.returned_->item_ == "allocated 1");
  BOOST_TEST(static_free_list.returned_->next_->item_ == "allocated 3");
  BOOST_TEST(!static_free_list.is_empty());
  BOOST_TEST(!static_free_list.is_full());
  BOOST_TEST(static_free_list.size_ == 2);

  static_free_list.remove(allocated_4);
  BOOST_TEST(allocated_4->next_->item_ == "allocated 1");
  BOOST_TEST(static_free_list.returned_->item_ == "allocated 4");
  BOOST_TEST(static_free_list.returned_->next_->item_ == "allocated 1");
  BOOST_TEST(!static_free_list.is_empty());
  BOOST_TEST(!static_free_list.is_full());
  BOOST_TEST(static_free_list.size_ == 1);

  static_free_list.remove(allocated_2);
  BOOST_TEST(allocated_2->next_->item_ == "allocated 4");
  BOOST_TEST(static_free_list.returned_->item_ == "allocated 2");
  BOOST_TEST(static_free_list.returned_->next_->item_ == "allocated 4");
  BOOST_TEST(static_free_list.is_empty());
  BOOST_TEST(!static_free_list.is_full());
  BOOST_TEST(static_free_list.size_ == 0);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(ConstructsForLargeValues)
{
  // Update: it works now because each element in nodes_ was not allocated
  // by using new. Each then gets deleted by either use of delete[] or calling
  // the dtor for each.
  // fatal error: signal: SIGABRT (application abort requested)
  Kedyk::StaticFreeList<std::string> static_free_list {42};
  BOOST_TEST(static_free_list.capacity_ == 42);
  BOOST_TEST(static_free_list.size_ == 0);
  BOOST_TEST(static_free_list.max_size_ == 0);
  BOOST_TEST(static_free_list.returned_ == nullptr);
  BOOST_TEST(static_free_list.is_empty());
  BOOST_TEST(!static_free_list.is_full());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(AllocateAllocates)
{
  // unknown location(0)
  Kedyk::StaticFreeList<std::string> static_free_list {42};

  auto* allocate_1 = static_free_list.allocate();

  BOOST_TEST_REQUIRE(static_free_list.returned_ == nullptr);
  BOOST_TEST(static_free_list.max_size_ == 1);
  BOOST_TEST(static_free_list.size_ == 1);

  // Obtain fatal error SIGABRT (application abort requested), unknown location
  // (0).
  allocate_1->item_ = "Tribe Called Quest";
  //static_free_list.nodes_[0].item_ = "Tribe Called Quest";

  //allocate_1->item_ = "Tribe Called Quest";

  //BOOST_TEST(static_free_list.nodes_[0].item_ == "Tribe Called Quest");
  BOOST_TEST(static_free_list.returned_ == nullptr);
  auto* allocate_2 = static_free_list.allocate(); 
  BOOST_TEST(static_free_list.max_size_ == 2);
  BOOST_TEST(static_free_list.size_ == 2);
  allocate_2->item_ = "De La Soul";
  BOOST_TEST(static_free_list.nodes_[1].item_ == "De La Soul");
  BOOST_TEST(static_free_list.returned_ == nullptr);

  auto* allocate_3 = static_free_list.allocate(); 
  BOOST_TEST(static_free_list.max_size_ == 3);
  BOOST_TEST(static_free_list.size_ == 3);
  allocate_3->item_ = "African Soul Child";
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
  Kedyk::StaticFreeList<std::string> static_free_list {4};
  BOOST_TEST(static_free_list.is_empty());
  BOOST_TEST(!static_free_list.is_full());
  BOOST_TEST(static_free_list.capacity_ == 4);


  auto* allocated_1 = static_free_list.allocate(); 
  BOOST_TEST(static_free_list.max_size_ == 1);
  BOOST_TEST(static_free_list.size_ == 1);

  auto* allocated_2 = static_free_list.allocate(); 
  BOOST_TEST(!static_free_list.is_empty());
  BOOST_TEST(!static_free_list.is_full());
  BOOST_TEST(static_free_list.max_size_ == 2);
  BOOST_TEST(static_free_list.size_ == 2);


  auto* allocated_3 = static_free_list.allocate(); 
  BOOST_TEST(!static_free_list.is_empty());
  BOOST_TEST(!static_free_list.is_full());
  BOOST_TEST(static_free_list.max_size_ == 3);
  BOOST_TEST(static_free_list.size_ == 3);

  auto* allocated_4 = static_free_list.allocate(); 
  BOOST_TEST(!static_free_list.is_empty());
  BOOST_TEST(static_free_list.is_full());
  BOOST_TEST(static_free_list.max_size_ == 4);
  BOOST_TEST(static_free_list.size_ == 4);
}

BOOST_AUTO_TEST_SUITE_END() // StaticFreeList_tests
BOOST_AUTO_TEST_SUITE_END() // Kedyk

BOOST_AUTO_TEST_SUITE_END() // DataStructures