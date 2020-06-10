//------------------------------------------------------------------------------
// \file Course_Schedule_II_210_tests.cpp
//------------------------------------------------------------------------------
#include <algorithm>
#include <boost/test/unit_test.hpp>
// cf. https://stackoverflow.com/questions/33644088/linker-error-while-building-unit-tests-with-boost
#include <boost/test/included/unit_test.hpp>
#include <iostream>
#include <vector>

BOOST_AUTO_TEST_SUITE(Algorithms)
BOOST_AUTO_TEST_SUITE(LeetCode)
BOOST_AUTO_TEST_SUITE(Course_Schedule_II_210_tests)

class Solution {
	public:
    std::vector<int> findOrder(
    	int numCourses,
    	std::vector<std::vector<int>>& prerequisites)
    {
    	std::vector<int> resulting_order;

			return resulting_order;
    }
};



//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DemonstrateGivenExample)
{

	BOOST_TEST(true);
}

BOOST_AUTO_TEST_SUITE_END() // Course_Schedule_II_210_tests
BOOST_AUTO_TEST_SUITE_END() // LeetCode
BOOST_AUTO_TEST_SUITE_END() // Algorithms