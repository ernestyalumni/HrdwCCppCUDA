/**
 * 	@file 	copyctordemo.cpp
 * 	@brief 	copy constructor is a member function which initializes an object using another object of the same class  
 * 	@ref	http://www.geeksforgeeks.org/copy-constructor-in-cpp/ 
 * 	@details 2 parts, 1, example class where copy constructor is needed, 
 * 2. and if we remove copy constructor; changes made to str2 reflect in str1 as well which is never expected      
 * 
 * COMPILATION TIP : -Wall warning all, -g debugger, gdb can step through 
 * g++ -Wall -g copyctordemo.cpp -o copyctordemo
 * */

#include <iostream>  
#include <cstring> 		// std::strlen  

/*
 * Write an example class where copy constructor is needed?  
 */

class StringCls
{
	private:
		char *s;
		int size;
	public:
		StringCls(const char *str = NULL); 	// constructor 
		~StringCls() { delete [] s; }		// destructor 
		StringCls(const StringCls&); 		// copy constructor
		void print() { std::cout << s << std::endl; }	// Function to print string
		void change(const char *); 			// Function to change 
};  

// constructor code
StringCls::StringCls(const char *str) 
{
	size = std::strlen(str); 
	s = new char[size + 1]; 
	std::strcpy(s, str);
}

// StringCls class member function  
void StringCls::change(const char *str) 
{
	delete [] s;
	size = std::strlen(str);
	s = new char[size+1];
	std::strcpy(s, str);
}

// copy constructor, code  
StringCls::StringCls(const StringCls& old_str)
{
	size = old_str.size;
	s = new char[size+1]; 
	std::strcpy(s, old_str.s);
}

/*
 * What would be the problem if we remove copy constructor from above code?  
 * 
 * @brief no copy constructor
 */ 
class StringNoCopy
{
	private:
		char *s;
		int size;
	public:
		StringNoCopy(const char *str= NULL);	// constructor 
		~StringNoCopy() { delete [] s; } 		// destructor
		void print() { std::cout << s << std::endl; }
		void change(const char *); 				// Function to change, i.e. public function to change private data 
};  

StringNoCopy::StringNoCopy(const char *str) 
{
	size = std::strlen(str); 
	s = new char[size+1]; 
	std::strcpy(s, str);
}

void StringNoCopy::change(const char *str) 
{
	delete [] s;
	size = std::strlen(str);
	s = new char[size+1]; 
	std::strcpy(s, str);
}

			

int main() 
{

	/*
	 * Write an example class where copy constructor is needed?  
	 */

	StringCls str1("GeeksQuiz");
	StringCls str2 = str1; 
	
	str1.print(); 	// what is printed ?	// GeeksQuiz 
	str2.print(); 	// GeeksQuiz
	
	str2.change("GeeksforGeeks");
	
	str1.print(); 	// what is printed now ?	// GeeksQuiz
	str2.print(); 	// GeeksforGeeks 

	/*
	 * What would be the problem if we remove copy constructor from above code?  
	 */ 
	StringNoCopy str1b("GeeksQuiz"); 
//	StringNoCopy str2b = str1b; // Aborted (core dumped)

	
	str1b.print(); 	// what is printed ? 	// GeeksQuiz
/*	str2b.print(); 
	
	str2b.change("GeeksforGeeks");
*/	
	str1b.print();	// what is printed now ?
/*	str2b.print();
*/

	return 0;
	
}
