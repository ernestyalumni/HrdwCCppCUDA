Boost Unit testing  

cf. [Introduction into testing or why testing is worth the effort](http://www.boost.org/doc/libs/1_45_0/libs/test/doc/html/tutorials/intro-in-testing.html)  

cf. [Hello the testing world or beginner's introduction into testing using the Unit Test Framework](http://www.boost.org/doc/libs/1_45_0/libs/test/doc/html/tutorials/hello-the-testing-world.html)

cf. [Explanation, Problem with Executing Boost Library](https://askubuntu.com/questions/104316/problem-with-executing-boost-library)  


i got the solution its order of the command instead of using
```  
g++ -o hello -lboost_unit_test_framework hello.cpp (which used to work in earlier version)
```  
change it to
```  
g++ -o hello hello.cpp -lboost_unit_test_framework
```  
This works fine. 

cf. `./addfunc_test.cpp`  
```  
#define BOOST_TEST_MODULE MyTest
#include <boost/test/unit_test.hpp>  

int add( int i, int j ) { return i+j; }  

BOOST_AUTO_TEST_CASE( my_test) 
{
	// seven ways to detect and report the same error: 
	BOOST_CHECK( add(2,2) == 4) ;		// #1 continues on error

	BOOST_REQUIRE( add( 2,2 ) == 4 ); 	// #2 throws on error 
	
	if ( add( 2,2 ) != 4) { 
		BOOST_ERROR( "Ouch..." ); 		// #3 continues on error 
	}
	
	if ( add( 2,2 ) != 4 ) {
		BOOST_FAIL( "Ouch..." ); 		// #4 throws on error
	} 
	
	if ( add( 2, 2) != 4 ) throw "Ouch..."; // #5 throws on error
	
	BOOST_CHECK_MESSAGE( add( 2,2 ) == 4, 	// #6 continues on error 
						"add(..) result:  " << add( 2, 2) ); 
			
	BOOST_CHECK_EQUAL( add( 2,2 ), 4); 		// #7 continues on error
}

int main(int argc, char* argv[]) 
{
//	BOOST_AUTO_TEST_CASE(my_test); // error, function definition
	
}
```  
1. `BOOST_CHECK` - displays error message (by default on `std::cout`) that includes expression that failed, source file name, and source file line number.  It also increments error count.  At program termination, error count displayed automatically by Unit Test Framework.  
2. `BOOST_REQUIRE` - similar to `BOOST_CHECK`, but after displaying error, exception is thrown, to be caught by Unit Test Framework.  This approach is suitable when writing an explicit test program, and error would be so severe as to make further testing impractical.  
`BOOST_REQUIRE` differs from C++ Standard Library's `assert()` macro in that it's always generated, and channels error detection into uniform Unit Test Framework reporting procedure.  
3. `BOOST_ERROR` - similar to `BOOST_CHECK` except error detection and error reporting coded separately.  This is most useful when specific condition being tested requires several independent statements and/or is not indicative of reason for failure.  
4. `BOOST_FAIL` - similar to `BOOST_REQUIRE` except error detection and error reporting coded separately.  Most useful when specific condition being tested requires several independent statements and/or is not indicative of reason for failure.  
5. `throw` throws an exception, which will be caught and reported by Unit Test Framework.  Error message displayed when exception is caught will be most meaningful if exception is derived from `std::exception`, or is `char*` or `std::string`.  
6. `BOOST_CHECK_MESSAGE` - similar to `BOOST_CHECK`, except displays alternative error message specified as 2nd. argument  
7. `BOOST_CHECK_EQUAL` - similar to `BOOST_CHECK`, except checks equality of 2 variables, since in case of error it shows mismatched values.  

cf. [Boost.Test driven development or "getting started" for TDD followers](http://www.boost.org/doc/libs/1_45_0/libs/test/doc/html/tutorials/new-year-resolution.html)   

```  
#define BOOST_TEST_MODULE stringtest
#include <boost/test/included/unit_test.hpp>
#include "./str.h" 

BOOST_AUTO_TEST_SUITE (stringtest) // name of the test suite is stringtest 

BOOST_AUTO_TEST_CASE (test1) 
{
	mystring s;
	BOOST_CHECK(s.size() == 0 ); 
}

BOOST_AUTO_TEST_CASE (test2) 
{
	mystring s;
	s.setbuffer("hello world"); 
	BOOST_REQUIRE_EQUAL ('h', s[0]); // basic test 
} 

BOOST_AUTO_TEST_SUITE_END( ) 
```  

`BOOST_AUTO_TEST_SUITE`, `BOOST_AUTO_TEST_SUITE_END` macros indicate start and end of the test suite, respectively.   
Individual tests reside between these macros, and in that sense their semantics are like C++ namespaces.  Each individual unit test is defined using `BOOST_AUTO_TEST_CASE` macro.  





