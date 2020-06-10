//------------------------------------------------------------------------------
// \file Surrounded_Regions_130_tests.cpp
//------------------------------------------------------------------------------
#include <algorithm>
#include <boost/test/unit_test.hpp>
// cf. https://stackoverflow.com/questions/33644088/linker-error-while-building-unit-tests-with-boost
#include <boost/test/included/unit_test.hpp>
#include <iostream>
#include <vector>

BOOST_AUTO_TEST_SUITE(Algorithms)
BOOST_AUTO_TEST_SUITE(LeetCode)
BOOST_AUTO_TEST_SUITE(Surrounded_Regions_130_tests)


// USE functors.
// cf. https://www.bogotobogo.com/cplusplus/functors.php
template <typename T>
class PrintElements
{
  public:
    void operator()(const T& element)
    {
      std::cout << element << ' ';
    }
};

class Solution {
	public:
	    void solve(std::vector<std::vector<char>>& board) {
	        
	    }
};

std::vector<bool> mark_adjacent_to_border_cell_from_border(
	const std::vector<char>& outer_border)
{
	auto is_O_on_border =
		[](const char& x) -> bool
		{
			return (x == 'O');
		};

	std::vector<bool> marked_border;

	std::transform(
		outer_border.begin(),
		outer_border.end(),
		std::back_inserter(marked_border),
		is_O_on_border);

	return marked_border;
};

// https://leetcode.com/problems/surrounded-regions/
const std::vector<std::vector<char>> given_board
{
	{'X', 'X', 'X', 'X'},
	{'X', 'O', 'O', 'X'},
	{'X', 'X', 'O', 'X'},
	{'X', 'O', 'X', 'X'}
};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(DemonstrateGivenExample)
{
	{
		std::for_each(
			given_board.begin()->begin(),
			given_board.begin()->end(),
			PrintElements<char>{});
	}



	BOOST_TEST(true);
}

BOOST_AUTO_TEST_SUITE_END() // 
BOOST_AUTO_TEST_SUITE_END() // LeetCode
BOOST_AUTO_TEST_SUITE_END() // Algorithms