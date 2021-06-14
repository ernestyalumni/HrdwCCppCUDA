#ifndef ALGORITHMS_PRINT_MATRIX_SPIRAL_H
#define ALGORITHMS_PRINT_MATRIX_SPIRAL_H

#include <cassert>
#include <vector>

namespace Algorithms
{

//------------------------------------------------------------------------------
/// \ref https://coderbyte.com/algorithm/print-matrix-spiral-order
//------------------------------------------------------------------------------
template <typename T>
class PrintMatrixSpiral
{
	public:

		using Vector = std::vector<T>;
		using Matrix = std::vector<Vector>;

		PrintMatrixSpiral() = default;

		//--------------------------------------------------------------------------
		/// \details June 13, 2021, 6:17 am stop. About 17 + 15 = 32 mins.
		/// O(M x N) time. Will traverse each element at least once.
		//--------------------------------------------------------------------------
		Vector get_linear_spiral_order(const Matrix& matrix)
		{
			assert(matrix.size() > 0);

			Vector linear_spiral_order;

			// M rows.
			const int M {static_cast<int>(matrix.size())};

			// N columns.
			const int N {static_cast<int>(matrix[0].size())};

			// Row bounds.
			int lower_row_bound {0};
			int upper_row_bound {M - 1};

			// Column bounds.
			int lower_column_bound {0};
			int upper_column_bound {N - 1};

			int current_row_ptr {0};
			int current_column_ptr {0};

			bool is_row_traversal {true};
			bool going_positive_row_direction {true};
			bool going_positive_column_direction {true};

			while (lower_row_bound <= upper_row_bound &&
				lower_column_bound <= upper_column_bound)
			{
				if (is_row_traversal)
				{
					if (going_positive_row_direction)
					{
						for (int j {lower_column_bound}; j <= upper_column_bound; ++j)
						{
							linear_spiral_order.emplace_back(matrix[current_row_ptr][j]);
						}

						lower_row_bound += 1;

						going_positive_row_direction = false;

						current_column_ptr = upper_column_bound;
					}
					else
					{
						for (int j {upper_column_bound}; j >= lower_column_bound; --j)
						{
							linear_spiral_order.emplace_back(matrix[current_row_ptr][j]);
						}

						upper_row_bound -= 1;

						going_positive_row_direction = true;

						current_column_ptr = lower_column_bound;
					}

					is_row_traversal = false;

				}
				else
				{
					if (going_positive_column_direction)
					{
						for (int i {lower_row_bound}; i <= upper_row_bound; ++i)
						{
							linear_spiral_order.emplace_back(matrix[i][current_column_ptr]);
						}

						upper_column_bound -= 1;

						going_positive_column_direction = false;

						current_row_ptr = upper_row_bound;
					}
					else
					{
						for (int i {upper_row_bound}; i >= lower_row_bound; --i)
						{
							linear_spiral_order.emplace_back(matrix[i][current_column_ptr]);
						}

						lower_column_bound += 1;

						going_positive_column_direction = true;

						current_row_ptr = lower_row_bound;
					}

					is_row_traversal = true;
				}
			}

			return linear_spiral_order;
		}

};

} // namespace Algorithms

#endif // ALGORITHMS_PRINT_MATRIX_SPIRAL_H