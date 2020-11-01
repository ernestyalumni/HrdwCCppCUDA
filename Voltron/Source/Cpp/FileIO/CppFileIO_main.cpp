//------------------------------------------------------------------------------
/// \file CppFileIO_main.cpp
/// \details
/// Example Usage
/// g++ -std=c++2a CppFileIO_main.cpp -o CppFileIO_mai
//------------------------------------------------------------------------------

#include <cstddef> // std::size_t
#include <cstdio> // std::FILE, std::fopen, std::remove
#include <cstdlib> // std::perror
#include <string> // std::stoi
#include <vector>

#include <fstream>
#include <sstream> // std::stringstream;
#include <iostream> // std:cerr

using std::size_t;
using std::FILE; 
using std::cerr;
using std::fopen;
//------------------------------------------------------------------------------
/// \url https://en.cppreference.com/w/cpp/string/basic_string/getline
/// \details
/// getline reads characters from an input stream and places them into a string.
/// default delimiter is endline character.
/// extracts characters from input istream and appends them to string str until
/// end of file, delim, or str.max_size() stored.
/// returns input, istream.
//------------------------------------------------------------------------------
using std::getline;
using std::ifstream;
using std::remove;
using std::string;
using std::stoi;
using std::vector;
using std::stringstream;

struct NameNumNum
{
	string name_;
	int x_;
	int y_;
};

int main()
{
  {
	// <cstdio>
	// std::FILE* fopen(const char* filename, const char* mode);

	//--------------------------------------------------------------------------
	/// What the data looks like for string int int .csv
	/// Method Man,1979,2044
	//--------------------------------------------------------------------------
  vector<NameNumNum> name_num_nums;

	const string base_directory {"../../../data/"};

	const string filename {"science.csv"};
	const string current_directory {"./"};
	const string fullpath {base_directory + filename};

	// No need to close an ifstream. RAII.
	// cf. https://stackoverflow.com/questions/748014/do-i-need-to-manually-close-an-ifstream
	ifstream file_in {fullpath};

	// cf. https://www.geeksforgeeks.org/csv-file-management-using-c/
	// Or do
	// fin.open("file.csv", ios::in);

	if (!file_in)
	{		
	 	cerr << "bad filename\n";
	 	cerr << "file name: " << fullpath << "\n";
	 	return -1;
	}

	// Read in and parse lines, while building up data structure.

	string line;
	// Or do while (file_in >> line)
	while (getline(file_in, line))
	{
		// size_t types
		//size_t 

		// position of number 1, start index.
		auto first_delimiter_position = line.find(',');
		auto num1_position = line.find(',') + 1;
		// position of number 2, start index.
		auto num2_position = line.find(',', num1_position) + 1;

		//--------------------------------------------------------------------------
		/// https://en.cppreference.com/w/cpp/string/basic_string/substr
		/// basic_string substr(size_type pos = 0, size_type count = npos);
		/// pos - position of the first character to include,
		/// count - length of the substring.
		/// Return string containing substring [pos, pos + count)
		//--------------------------------------------------------------------------

		auto name = line.substr(0, first_delimiter_position);

		// Expect type int from string to int conversion.
		auto num1 =
			stoi(line.substr(num1_position, num2_position - num1_position -1));

		auto num2 = stoi(line.substr(num2_position));

		name_num_nums.emplace_back(NameNumNum{name, num1, num2});

		// cf. https://www.geeksforgeeks.org/csv-file-management-using-c/
		// or do
		// string temp;
		// vector<string> row;
		// getline(file_in, temp);
		// Used for breaking words.
		// stringstream s {temp};

		// Read every column data of a row and store it in a string variable word.
		// string word; ',' is the delimiter to stop at.
		// while (getline(s, word, ','))
		// { add all the column data of a row to a vector,
		// row.push_back(word); } // row now has all data of that line.

	}


	if (name_num_nums.size() > 5)
	{
		for (int i {0}; i < 5; ++i)
		{
			std::cout << name_num_nums.at(i).name_ << " " <<
				name_num_nums.at(i).x_ << " " << name_num_nums.at(i).y_ << "\n";
		}
	}

	std::cout << "\nTotal size of file lines: " << name_num_nums.size() << "\n";

	/*
	//
	// If file is not there, failure to open and fclose would cause segmentation fault.
	//FILE* fp =  fopen(fullpath.data(), "r");
	// "w", write, Create a file for writing, destroy contents if file already
	// exists, create new file if it does not exist.
	FILE* fp =  fopen(fullpath.data(), "w");

	if (!fp)
	{
	  std::perror("File opening failed");
	}


	fclose(fp);

	// Delete file.
	// int remove(const char* fname)
	const int remove_result {remove(fullpath.data())};
	
  }
  */
  }

  {
  	vector<vector<string>> rows;

		const string base_directory {"../../../data/"};

		const string filename {"uk_ecommerce_data.csv"};
		const string fullpath {base_directory + filename};

		ifstream file_in {fullpath};

		if (!file_in)
		{
			cerr << "bad filename : " << fullpath << "\n";
		}

		string line;
		while (getline(file_in, line))
		{
			vector<string> row;
			size_t start_position {0};
			auto delimiter_position = line.find(',');
			auto next_position = delimiter_position;

			while (next_position != string::npos)
			{
				row.emplace_back(
					line.substr(start_position, next_position - start_position));

				start_position = next_position + 1;
				next_position = line.find(',', start_position);
			}

			rows.emplace_back(row);
		}

		std::cout << "\n Total size of file lines 2: " << rows.size() << "\n";

		if (rows.size() > 10)
		{
			for (int i {0}; i < 10; ++i)
			{
				for (int j {0}; j < rows.at(i).size(); ++j)
				{
					std::cout << rows.at(i).at(j) << " ";
				}
				std::cout << " size of row: " << rows.at(i).size() << "\n";
			}
		}
  }
  {
  	vector<vector<string>> rows;

		const string base_directory {"../../../data/"};

		const string filename {"uk_ecommerce_data.csv"};
		const string fullpath {base_directory + filename};

		ifstream file_in {fullpath};

		if (!file_in)
		{
			cerr << "bad filename : " << fullpath << "\n";
		}

		string temp;
		while (getline(file_in, temp))
		{
			vector<string> row;
			stringstream str_stream {temp};
			string word;

			while (getline(str_stream, word, ','))
			{
				row.emplace_back(word);
			}

			rows.emplace_back(row);
		}

		std::cout << "\n Total size of file lines 2: " << rows.size() << "\n";

		if (rows.size() > 10)
		{
			for (int i {0}; i < 10; ++i)
			{
				for (int j {0}; j < rows.at(i).size(); ++j)
				{
					std::cout << rows.at(i).at(j) << " ";
				}
				std::cout << " size of row: " << rows.at(i).size() << "\n";
			}
		}
  }

}