/**
 * @file   : parameterpack.cpp
 * @brief  : Parameter pack C++11, examples, i.e. args&& ...   
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20180124
 * @ref    : http://en.cppreference.com/w/cpp/language/parameter_pack
 * https://eli.thegreenplace.net/2014/variadic-templates-in-c/
 * @details : 
 * 
 * If you find this code useful, feel free to donate directly and easily at this direct PayPal link: 
 * 
 * https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=ernestsaveschristmas%2bpaypal%40gmail%2ecom&lc=US&item_name=ernestyalumni&currency_code=USD&bn=PP%2dDonationsBF%3abtn_donateCC_LG%2egif%3aNonHosted 
 * 
 * which won't go through a 3rd. party such as indiegogo, kickstarter, patreon.  
 * Otherwise, I receive emails and messages on how all my (free) material on 
 * physics, math, and engineering have helped students with their studies, 
 * and I know what it's like to not have money as a student, but love physics 
 * (or math, sciences, etc.), so I am committed to keeping all my material 
 * open-source and free, whether or not 
 * sufficiently crowdfunded, under the open-source MIT license: 
 * 	feel free to copy, edit, paste, make your own versions, share, use as you wish.  
 *  Just don't be an asshole and not give credit where credit is due.  
 * Peace out, never give up! -EY
 * 
 * */
/* 
 * COMPILATION TIP
 * g++ -std=c++14 -pthread thread2.cpp -o thread2
 * 
 * */
#include <iostream>

#include <typeinfo> // typeid

void tprintf(const char* format) // base function
{
	std::cout << format;
}

template<typename T, typename... Targs> 
void tprintf(const char* format, T value, Targs... Fargs) // recursive variadic function
{
	for ( ; *format != '\0'; format++ ) {
		if ( *format == '%' ) {
			std::cout << value; 
			tprintf(format+1, Fargs...); // recursive call
			return; 
		} 
		std::cout << *format; 
	}
}

/**
 * @ref https://eli.thegreenplace.net/2014/variadic-templates-in-c/
 * @details When compiler puts the whole program together, it does know.  
 * It sees perfectly well all the invocations of the function throughout the program, 
 * and all possible argument types it gets passed (types are, after all, resolved at compile-time in C++) 
 * @brief  function that adds all of its arguments together:
 */
template<typename T>
T adder(T v) {
	return v;
}

/**
 * @arg typename .. Args is called a template parameter pack and 
 * Args... args is called a function parameter pack 
 * (Args is, of course, a completely arbitrary name and could be anything else).  
 * 
 * Variadic templates are written just the way you'd write recursive code - 
 * you need a base case (the adder(T v) declaration above) and a 
 * general case which "recurses" 
 * 
 * The recursion itself happens in the call adder(args...). 
 * */
template<typename T, typename... Args>
T adder(T first, Args... args) {
	return first + adder(args...);
}

/**
 * @details use __PRETTY_FUNCTION__ macro 
 * */
template<typename T>
T adder_PRETTYPRINT(T v) {
	std::cout << __PRETTY_FUNCTION__ << "\n"; 
	return v;
}

template<typename T, typename... Args>
T adder_PRETTYPRINT(T first, Args... args) {
	std::cout << __PRETTY_FUNCTION__ << "\n"; 
	return first + adder(args...); 
}

// Type-safe variadic functions
// Variadic data structures 
// 
/** 
 * @ref https://eli.thegreenplace.net/2014/variadic-templates-in-c/
 * @brief This wasn't possible prior to C++11
 * Custom data structures (structs since times of C, classes in C++) have compile-time defined fields.   
 * They can represent types that grow at runtime (std::vector, e.g.) but if you want to add new fields, 
 * this is something compiler has to see.  
 * Variadic templates make it possible to 
 * define data structures that could have an arbitrary number of fields, and 
 * have this number configured per use.  
 * @ref https://github.com/eliben/code-for-blog/blob/master/2014/variadic-tuple.cpp
 * */

/* base case, def. of class template named tuple, which is empty 
 * 
 * */
template <class... Ts> struct tuple {};

/* this specialization peels off 1st type from parameter pack, and 
 * defines member of that type named tail.  
 * derives from the tuple instantiated with rest of the pack.  
 * This is a recursive definition that stops when there's no more types to peel off, and 
 * base of hierarchy is empty tuple.  
 * */
template <class T, class... Ts>
struct tuple<T, Ts...> : tuple<Ts...> { 
	tuple(T t, Ts... ts) : tuple<Ts...>(ts...), tail(t) {}
	
	T tail;
};

template <size_t, class> struct elem_type_holder;

// the way to access tuples is with get function template, 
template <class T, class... Ts>
struct elem_type_holder<0, tuple<T, Ts...>> {
	typedef T type;
};

/* elem_type_holder : <k, tuple> |-> elem_type_holder<k, tuple>
 * Note that this is compile-time template metaprogramming construct - 
 * it acts on constant and types, not on runtime objects.  
 * */
template <size_t k, class T, class... Ts>
struct elem_type_holder<k, tuple<T, Ts...>> {
	typedef typename elem_type_holder<k - 1, tuple<Ts...>>::type type;
};

// armed with elem_type_holder, implement get
template <size_t k, class... Ts>
typename std::enable_if<
	k == 0, typename elem_type_holder<0, tuple<Ts...>>::type&>::type
get(tuple<Ts...>& t) {
	return t.tail;
}

template <size_t k, class T, class... Ts>
typename std::enable_if<
	k != 0, typename elem_type_holder<k, tuple<T, Ts...>>::type&>::type
get(tuple<T, Ts...>& t) {
	tuple<Ts...>& base = t;
	return get<k - 1>(base);
}


int main() 
{
	tprintf("% world% %\n", "Hello", '!', 123); 

	/**
	 * @details adder will accept any number of arguments, and 
	 * will compile properly as long as it can apply + operator to them.  
	 * This checking is done by compiler, at compile time.  
	 * There's nothing magical about it - it follows C++'s usual template and overload resolution rules.  
	 * @ref https://eli.thegreenplace.net/2014/variadic-templates-in-c/
	 * */

	long sum = adder(1,2,3,8,7) ; 
	std::string s1 = "x", s2 = "aa", s3 = "bb", s4 = "yy"; 
	std::string ssum = adder(s1, s2, s3, s4);  

	/* sanity check */
	std::cout << " sum : " << sum << std::endl; 
	std::cout << " ssum : " << ssum << std::endl;  

	adder_PRETTYPRINT(1,2,3,8,7);

	tuple<double, uint64_t, const char*> t1(12.2, 42, "big");

	std::cout << "0th elem is " << get<0>(t1) << "\n";
	std::cout << "1th elem is " << get<1>(t1) << "\n"; 
	std::cout << "2th elem is " << get<2>(t1) << "\n"; 
	
	get<1>(t1) = 103; 
	std::cout << "1th elem is " << get<1>(t1) << "\n"; 
	
	typename elem_type_holder<1, tuple<double, int, const char*>>::type foo;
	typename elem_type_holder<2, tuple<double, int, const char*>>::type foo2;
	typename elem_type_holder<0, tuple<double, int, const char*>>::type foo0;

	std::cout << typeid(foo).name() << "\n"; 	// i 
	std::cout << typeid(foo2).name() << "\n";	// PKc
	std::cout << typeid(foo0).name() << "\n";	// d 
	

	return 0;
}

