//------------------------------------------------------------------------------
/// \file MainCommandLineArguments.cpp
/// \author Ernest Yeung
/// \brief Main file to test out command line arguments.
///
/// \details int main() is needed.
/// \ref Bjarne Stroustrup. The C++ Programming Language, 4th Edition.
/// Addison-Wesley Professional. May 19, 2013. ISBN-13: 978-0321563842. 
/// cf. pp. 253, 10.2.7 Command-Line Arguments, Stroustrup (2013)
///-----------------------------------------------------------------------------
#include <iostream>
#include <sstream>

int main(int argc, char* argv[])
{
	std::cout << "\n argc : " << argc << '\n';

	std::stringstream ss;

	for (int argument {0}; argument < argc; ++argument)
	{
		std::cout << " argument number : " << argument << " argv[argument] : " <<
			std::string{argv[argument]} << "; \n";

		// If you *don't* include this " ", then the strings "run into each other",
		// i.e. it's just one long string to stringstream's perspective, e.g.
		// ./CommandLineArguments --sf --se results in
		// ./CommandLineArguments--sf--se;
		// so instead of ss << argv[argument]; do this:
		ss << argv[argument] << " ";
	}

	std::cout << "\n From stringstream : \n";

	for (int argument {0}; argument < argc; ++argument)
	{
		std::string string_output;
		ss >> string_output;
		std::cout << " argument number : " << argument << " string output : " <<
			string_output << "; \n";
	}

}