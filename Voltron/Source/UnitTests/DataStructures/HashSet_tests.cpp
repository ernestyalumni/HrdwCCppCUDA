//------------------------------------------------------------------------------
/// \file HashSet_tests.cpp
/// \date 20201029 12:05
//------------------------------------------------------------------------------
#include "DataStructures/HashSet.h"

#include <boost/test/unit_test.hpp>
#include <memory> // std::make_shared

using DataStructures::HashTables::HashSet::HashSetT;
using DataStructures::HashTables::HashSet::Node;
using DataStructures::HashTables::HashSet::NodeShared;
using std::make_shared;
using std::shared_ptr;

BOOST_AUTO_TEST_SUITE(DataStructures)
BOOST_AUTO_TEST_SUITE(HashSet_tests)


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(HashSetNodeConstructs)
{
  {
    // Dynamically allocate because you don't know the size at compile time. For
    // example std::vector, you don't know the size at runtime, internally it's
    // dynamically allocated.
    // Static allocation is deallocated as soon as the allocating function
    // returns.
    // Returns a pointer to the beginning of the new block of memory.
    Node<int>* root_ptr {new Node<int>{42}};

    Node<int>* left_ptr {new Node<int>{7}};
    Node<int>* right_ptr {new Node<int>{69}};

    root_ptr->left_ = left_ptr;
    root_ptr->right_ = right_ptr;

    BOOST_TEST(root_ptr->left_->value_ == 7);
    BOOST_TEST(root_ptr->right_->value_ == 69);
    BOOST_TEST(root_ptr->left_->left_ == nullptr);
    BOOST_TEST(root_ptr->left_->right_ == nullptr);
    BOOST_TEST(root_ptr->right_->left_ == nullptr);
    BOOST_TEST(root_ptr->right_->right_ == nullptr);

    // Given pointer to memory block previously allocated with new, free up
    // memory.
    delete root_ptr;
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(HashSetNodeSharedConstructsWithMakeShared)
{
  {
    // cf. https://www.acodersjourney.com/top-10-dumb-mistakes-avoid-c-11-smart-pointers/
    // make_shared dynamically allocates for manager and new object at same
    // time. i.e. When you use make_shared, C++ compiler does a single memory
    // allocation big enough to hold both manager object and the new object.
    shared_ptr<NodeShared<int>> root_ptr {make_shared<NodeShared<int>>(42)};

    // Wrong, there are 2 dynamic memory allocations that happen, one for the
    // object itself from new, and then second for the manager object created by
    // the shared_ptr ctor.
    // shared_ptr<Node<int>> root_ptr {new Node<int>{42}};

    shared_ptr<NodeShared<int>> left_ptr {make_shared<NodeShared<int>>(7)};
    shared_ptr<NodeShared<int>> right_ptr {make_shared<NodeShared<int>>(69)};

    root_ptr->left_ = left_ptr;
    root_ptr->right_ = right_ptr;

    BOOST_TEST(root_ptr->value_ == 42);
    BOOST_TEST(root_ptr->left_->value_ == 7);
    BOOST_TEST(root_ptr->right_->value_ == 69);
    BOOST_TEST(root_ptr->left_->left_ == nullptr);
    BOOST_TEST(root_ptr->left_->right_ == nullptr);
    BOOST_TEST(root_ptr->right_->left_ == nullptr);
    BOOST_TEST(root_ptr->right_->right_ == nullptr);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(HashSetTIsASet)
{
  HashSetT<int>* obj = new HashSetT<int>{};

  obj->add(1);
  obj->add(2);
  BOOST_TEST(obj->contains(1));
  BOOST_TEST(!(obj->contains(3)));
  obj->add(2);
  BOOST_TEST(obj->contains(2));
  obj->remove(2);
  BOOST_TEST(!obj->contains(2));
}

BOOST_AUTO_TEST_SUITE_END() // HashSet_tests
BOOST_AUTO_TEST_SUITE_END() // DataStructures