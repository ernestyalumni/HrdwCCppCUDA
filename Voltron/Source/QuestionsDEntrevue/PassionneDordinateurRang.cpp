//------------------------------------------------------------------------------
/// \file PassionneDordinateurRang.cpp
/// \author Ernest Yeung
/// \brief
/// \ref 
///-----------------------------------------------------------------------------

#include <algorithm> // std::find_if, std::max, std::min
#include <cctype> // std::isspace
#include <cstddef> // std::size_t 
#include <functional> // std::not_fn, std::ref
#include <iterator> // std::back_inserter;
#include <string>
#include <vector>

using std::back_inserter;
using std::find_if;
using std::function;
using std::isspace;
using std::max;
using std::min;
using std::not_fn;
using std::ref;
using std::size_t;
using std::string;
using std::transform;
using std::vector;

namespace QuestionsDEntrevue
{

namespace PassionneDordinateurRang
{

namespace ActivezLesFontaines
{

// range from position max((i - locations[i]), 1) to min((i + locations[i]), n)

vector<int> compute_left_ranges(vector<int>& emplacements)
{
	vector<int> left_range_values;

	//auto compute_left_range = [](const int )

	//transform(emplacements.begin(), emplacements.end(),
	//back_inserter(left_range_values));

	for (size_t index {0}; index < emplacements.size(); ++index)
	{
		const int i {static_cast<int>(index) + 1}; // 1-based counting for i.

		const int result {max((i - emplacements.at(index)), 1)};

		left_range_values.emplace_back(result);
	}

	return left_range_values;
}

vector<int> compute_right_ranges(vector<int>& emplacements)
{
	vector<int> right_range_values;

	//transform(emplacements.begin(), emplacements.end(),
	//back_inserter(right_range_values));

	for (size_t index {0}; index < emplacements.size(); ++index)
	{
		const size_t i {index + 1}; // 1-based counting for i.

		right_range_values.emplace_back(
			min((i + emplacements.at(index)), emplacements.size()));
	}

	return right_range_values;
}

vector<int> count_fountains_by_memo(vector<int>& emplacements)
{
	// Indices (0-based counting) of fountains needed for covering garden.
	vector<int> fountains_needed;

	const vector<int> left_range_values {compute_left_ranges(emplacements)};
	const vector<int> right_range_values {compute_right_ranges(emplacements)};

	// 0-based counting
	int left_most_index_tracker {0};
	int right_most_index_tracker {0};

	for (size_t i {0}; i < emplacements.size(); ++i)
	{
		// O(1) since access by index for an array is O(1). 0 based counting.
		const int left_range {left_range_values[i] - 1};
		const int right_range {right_range_values[i] - 1};

		if (right_most_index_tracker < right_range)
		{
			if (left_range <= left_most_index_tracker)
			{
				vector<int> new_result;

				for (int fountain_index : fountains_needed)
				{
					if (left_range_values[fountain_index] < left_range)
					{
						new_result.emplace_back(fountain_index);
					}
				}

				fountains_needed = new_result;
			}

			right_most_index_tracker = right_range;
			left_most_index_tracker = left_range;

			fountains_needed.emplace_back(i);
		}
	}
	return fountains_needed;
}

string couper_gauche(const string& str)
{
	string s {str};

	s.erase(
		s.begin(),
		find_if(
			s.begin(),
			s.end(),
			// template <class F>
			// not_fn(F&& f);
			// int isspace(int ch)
			not_fn<int(*)(int)>(isspace)));
			// Alternatively, use a lambda function,
			// [](auto input_char) { return !isspace(input_char); }

	return s;
}

} // namespace ActivezLesFontaines

} // namespace PassionneDordinateurRang

} // namespace QuestionsDEntrevue

