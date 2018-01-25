/**
 * @file   : explicit.cpp
 * @brief  : explicit keyword C/C++11, constructor   
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20180124
 * @ref    : http://en.cppreference.com/w/cpp/language/explicit 
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
 * g++ -std=c++14 explicit.cpp -o explicit
 * 
 * */
 
struct A
{
	A(int) { } 			// converting constructor
	A(int, int) { } 	// converting constructor (C++11)
	operator bool() const { return true; }
};

struct B
{
	explicit B(int) { }
	explicit B(int, int) { }
	explicit operator bool() const { return true; }
}; 

int main() 
{
	A a1 = 1;		// OK: copy-initialization selects A::A(int)
	A a2(2); 		// OK: direct-initialization selects A::A(int)
	A a3 { 4, 5}; 	// OK: direct-list-initialization selects A::A(int, int)
	A a4 = {4, 5}; 	// OK: copy-list-initialization selects A::A(int, int)
	A a5 = (A)1; 	// OK: explicit cast performs static_cast 
	if (a1) ;		// OK: A::operator bool()
	bool na1 = a1; 	// OK: copy-initialization selects A::operator bool()
	bool na2 = static_cast<bool>(a1); 	// OK: static_cast performs direct-initialization 
	
// 	B b1 = 1;		// error: copy-initialization does not consider B::B(int)
	B b2(2);		// OK: direct-initialization selects B::B(int) 
	B b3 {4, 5};	// OK: direct-list-initialization selects B::B(int, int)
// B b4 = {4, 5}; 	// error: copy-list-initialization does not consider B::B(int, int)
	B b5 = (B)1;	// OK: explicit cast performs static_cast
	if (b2) ; 		// OK: B:operator bool()
// bool nb1 = b2;	// error: copy-initialization does not consider B::operator bool()
	bool nb2 = static_cast<bool>(b2); 	// OK : static_cast performs direct-initialization
}	
	
 
