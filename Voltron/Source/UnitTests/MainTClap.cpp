//------------------------------------------------------------------------------
/// \file MainTClap.cpp
/// \author Ernest Yeung
/// \brief Main file to test out TCLAP.
///
/// \details int main() is needed.
/// \ref http://tclap.sourceforge.net/manual.html#EXAMPLE
///-----------------------------------------------------------------------------
#include <algorithm> // std::reverse
#include <iostream>
#include <sstream>
#include <string>
#include <tclap/CmdLine.h>

int main(int argc, char* argv[])
{
	// Wrap everything in a try block. Do this every time, because
	// exceptions will be thrown for problems.

	try
	{
		// Define the command line object, and
		// insert a message that describes the program.
		// The "Command description message" is printed last im the help text.
		// The second argument is the delimiter (usually space) and the
		// last one is the version number.
		// The CmdLine object parses the argv array based on the Arg objects, that
		// it contains.
		//
		// cf. http://tclap.sourceforge.net/manual.html#COMMAND_LINE
		// CmdLine class contains arguments that define the command line and manages
		// parsing of command line.
		// Actual parsing of individual arguments occurs within arguments
		// themselves.
		// CmdLine keeps track of required arguments, relationships between
		// arguments, and output generation.
		// cf. http://tclap.sourceforge.net/html/classTCLAP_1_1CmdLine.html#a6ee7bc32fd24e6e6f966b32f03df9124
		// TCLAP::CmdLine::CmdLine(const str::string& message,
		// const char delimiter = ' ',
		// const std::string& version = "none",
		// bool helpAndVersion = true);
		// delimiter - character that's used to separate argument flag/name from
		// value. Defaults to ' ' (space)
		TCLAP::CmdLine cmd {"Command description message", ' ', "0.9"};

		// Define a value argument and add it to the command line.
		// A value arg defines a flag and a type of value that it expects, such as
		// "-n Bishop".
		//
		// http://tclap.sourceforge.net/html/classTCLAP_1_1ValueArg.html
		// basic labeled argument that parases a value.
		// http://tclap.sourceforge.net/html/classTCLAP_1_1ValueArg.html#ab2818435a1749bee3233b1f364dabf09
		// TCLAP::ValueArg<T>::ValueArg(const std::string& flag,
		// const std::string& name,
		// const std::string& desc,
		// bool req,
		// T value,
		// const std::string& typeDesc,
		// Visitor* v = NULL)
		// flag - the one character flag that identifies this argument on the
		// command line.
		// name - A one word name for the argument. Can be used as a long flag on
		// the command line.
		// desc - A description of what argument is for or does
		// req - Whether the argument is required on the command line.
		// value - The default value assigned to this argument if it's not present
		// on the command line
		// typeDesc - A short, human readable description of the type that this
		// object expected. This is used in generation of the USAGE statement.
		// v - An optional visitor. You probably shouldn't use this unless you have
		// a very good reason.
		TCLAP::ValueArg<std::string> nameArg {
			"n",
			"name",
			"Name to print",
			true,
			"homer",
			"string"};

		// Add the argument nameArg to the CmdLine object.
		// The CmdLine object uses this Arg to parse the command line.
		cmd.add(nameArg);

		// Define a switch and add it to the command line.
		// A switch arg is a boolean argument and only defines a flag that indicates
		// true or false.
		// In this example the SwitchArg adds itself to the CmdLine object as part
		// of the constructor. This eliminates the need to call the cmd.add()
		// method. All args have support in their constructors to add themselves
		// directly to the CmdLine object.
		// It doesn't matter which idiom you choose, they accomplish the same thing.
		TCLAP::SwitchArg reverseSwitch {
			"r", "reverse", "Print name backwards", cmd, false};

		// Parse the argv array.
		cmd.parse(argc, argv);

		// Get the value parsed by each.
		std::string name {nameArg.getValue()};
		const bool reverseName {reverseSwitch.getValue()};

		std::cout << "\n nameArg.getValue (name) : " << name << '\n';
		std::cout << "\n reverseSwitch.getValue() (reverseName) : " <<
			reverseName << '\n';

		// Do what you intend.
		if (reverseName)
		{
			std::reverse(name.begin(), name.end());
			std::cout << "My name (spelled backwards) is: " << name << std::endl;
		}
		else
		{
			std::cout << "My name is: " << name << std::endl;
		}
	}
	catch (TCLAP::ArgException &e) // catch any exceptions
	{
		std::cerr << "error: " << e.error() << " for arg " << e.argId() <<
			std::endl;
	}
}