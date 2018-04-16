/**
 * @file   : classes.cpp
 * @author : Ernest Yeung
 * @email  : ernestyalumni@gmail.com
 * @brief  : Program to demonstrate classes.
 * @ref    : Ch. 16 Classes; Bjarne Stroustrup, The C++ Programming Language, 4th Ed., Stroustrup  
 * If you find this code useful, feel free to donate directly and easily at this direct PayPal link: 
 * 
 * https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=ernestsaveschristmas%2bpaypal%40gmail%2ecom&lc=US&item_name=ernestyalumni&currency_code=USD&bn=PP%2dDonationsBF%3abtn_donateCC_LG%2egif%3aNonHosted 
 * 
 * which won't go through a 3rd. party such as indiegogo, kickstarter, patreon.  
 * Otherwise, I receive emails and messages on how all my (free) material on physics, math, and engineering have 
 * helped students with their studies, and I know what it's like to not have money as a student, but love physics 
 * (or math, sciences, etc.), so I am committed to keeping all my material open-source and free, whether or not 
 * sufficiently crowdfunded, under the open-source MIT license: 
 * 	feel free to copy, edit, paste, make your own versions, share, use as you wish.    
 * Peace out, never give up! -EY
 * 
 * */
#include <iostream> 

class X {
    private:        // the representation (implementation) is private 
        int m;
    public:                     // the user interface is public
        X(int i =0) : m {i} {} // a constructor (initialize the data member m)

        int mf(int i)           // a member function
        {
            int old = m;
            m = i;      // set a new value
            return old; // return the old value 
        }
}; 

int user(X var, X* ptr)
{
    int x = var.mf(7);      // access using . (dot)
    int y = ptr->mf(9);     // access using -> (arrow)
//    int z = var.m;          // error: cannot access private member
}


int main(int argc, char* argv[])
{
    X var {7};  // a variable of type X, initialized to 7
}
