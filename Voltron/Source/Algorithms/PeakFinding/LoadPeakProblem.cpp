//------------------------------------------------------------------------------
/// \file LoadPeakProblem.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief Load the 2-dim. problem for peak finding of Problem Set 1, MIT 6.006.
/// \details MIT OCW 6.006, Fall 2011, Problem Set 1.
//------------------------------------------------------------------------------
#include "LoadPeakProblem.h"

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <iterator> // std::back_inserter
#include <optional>
#include <sstream>
#include <string> // std::getline, std::stoi
#include <utility> // std::move, std::pair
#include <vector>

using std::back_inserter;
using std::getline;
using std::ifstream;
using std::istringstream;
using std::make_optional;
using std::make_pair;
using std::move;
using std::nullopt;
using std::optional;
using std::pair;
using std::size_t;
using std::stoi;
using std::string;
using std::transform;
using std::vector;

namespace Algorithms
{
namespace PeakFinding
{

LoadPeakProblem::LoadPeakProblem() = default;

void LoadPeakProblem::parse(ifstream& input_file_stream)
{
	for (string temp_line; getline(input_file_stream, temp_line);)
	{
		const auto resulting_boundaries = get_row_boundaries(temp_line);

		if (resulting_boundaries)
		{
			const auto parsed_result =
				simple_parse_row(temp_line, *resulting_boundaries);

			vector<int> transformed_result;

			// cf. https://stackoverflow.com/questions/33522742/replacing-stdtransform-inserting-into-a-stdvector
			// std::back_inserter It will act like an iterator to the vector but will
			// actually call push_back() to insert the elements into the vector.
			transform(
				parsed_result.begin(),
				parsed_result.end(),
				back_inserter(transformed_result),
				[](auto input) {
					return stoi(input); });	
		
			data_.emplace_back(transformed_result);
		}
	}
}

optional<pair<size_t, size_t>> LoadPeakProblem::get_row_boundaries(
	const string& input_line)
{
	size_t find_left_bracket {input_line.find('[')};
	size_t find_right_bracket {input_line.find(']')};

	return (find_left_bracket == string::npos || find_right_bracket == string::npos) ?
		nullopt : make_optional<pair<size_t, size_t>>(make_pair<size_t, size_t>(
			move(find_left_bracket),
			move(find_right_bracket)));
}

vector<string> LoadPeakProblem::simple_parse_row(
	const string& input_line,
	const pair<size_t, size_t>& row_boundaries)
{
	const size_t find_left_bracket {row_boundaries.first};
	const size_t find_right_bracket {row_boundaries.second};

	const string row_data_string {
		input_line.substr(
			find_left_bracket + 1,
			find_right_bracket - find_left_bracket - 1)};

	istringstream iss {row_data_string};

	string output_str;

	vector<string> output;

	while (getline(iss, output_str, ','))
	{
		output.emplace_back(output_str);
	}

	return output;
}

optional<size_t> LoadPeakProblem::parse_first_equal_sign(const string& input_line)
{
	const size_t result {input_line.find('=')};
	return result == string::npos ? nullopt : make_optional<size_t>(result);
}

pair<optional<size_t>, optional<size_t>> LoadPeakProblem::parse_row_brackets(
	const string& input_line)
{
	size_t find_left_bracket {input_line.find('[')};
	size_t find_right_bracket {input_line.find(']')};

	optional<size_t> left_bracket_position {
		find_left_bracket == string::npos ? nullopt :
			make_optional<size_t>(find_left_bracket)};
	optional<size_t> right_bracket_position {
		find_right_bracket == string::npos ? nullopt :
			make_optional<size_t>(find_right_bracket)};

	if (left_row_bracket_found_)
	{
		if (static_cast<bool>(right_bracket_position))
		{
			if (static_cast<bool>(left_bracket_position))
			{
				// [ ... ]
				if (*left_bracket_position < *right_bracket_position)
				{
					return make_pair<optional<size_t>, optional<size_t>>(
						move(left_bracket_position),
						move(right_bracket_position));
				}
				// ] ... [ ...
				else
				{
					return make_pair<optional<size_t>, optional<size_t>>(
						nullopt,
						move(right_bracket_position));
				}
			}
			// ] ...
			else
			{
				left_row_bracket_found_ = false;
				return make_pair<optional<size_t>, optional<size_t>>(
					nullopt,
					move(right_bracket_position));
			}
		}
		// [ ...
		else
		{
			return make_pair<optional<size_t>, optional<size_t>>(
				move(left_bracket_position),
				nullopt);
		}
	}
	else
	{
		if (static_cast<bool>(right_bracket_position))
		{
			if (static_cast<bool>(left_bracket_position))
			{
				// [ ... ]
				if (*left_bracket_position < *right_bracket_position)
				{
					return make_pair<optional<size_t>, optional<size_t>>(
						move(left_bracket_position),
						move(right_bracket_position));
				}
				// ] ... [ ...
				else
				{
					return make_pair<optional<size_t>, optional<size_t>>(nullopt, nullopt);
				}
			}
			// ] ...
			else
			{
				left_row_bracket_found_ = false;
				return make_pair<optional<size_t>, optional<size_t>>(nullopt, nullopt);
			}
		}
		// [ ...
		else
		{
			left_row_bracket_found_ = true;
			return make_pair<optional<size_t>, optional<size_t>>(
				move(left_bracket_position),
				nullopt);
		}
	}
}

} // namespace PeakFinding
} // namespace Algorithms
