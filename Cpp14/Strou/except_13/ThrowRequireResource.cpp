/**
 * @file   : ThrowRequireResource.cpp
 * @author : Ernest Yeung
 * @email  : ernestyalumni@gmail.com
 * @brief  : delete before a throw or leak
 * @ref    : 13.2 Exception Guarantees Ch. 13 Exception Handling; 
 *   Bjarne Stroustrup, The C++ Programming Language, 4th Ed., Stroustrup  
 * @detail Operation that throws an exception must not only leave its operands 
 *   in well-defined states, but must also ensure that every resource it 
 *   acquired is (eventually) released. e.g. at the point where an exception is
 *   thrown, all memory allocated must be either deallocated or owned by some 
 *   object, which in turn must ensure that memory is properly deallocated. 
 * 
 * If you find this code useful, feel free to donate directly and easily at this direct PayPal link: 
 * 
 * https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=ernestsaveschristmas%2bpaypal%40gmail%2ecom&lc=US&item_name=ernestyalumni&currency_code=USD&bn=PP%2dDonationsBF%3abtn_donateCC_LG%2egif%3aNonHosted 
 * 
 * which won't go through a 3rd. party such as indiegogo, kickstarter, patreon.  
 * Otherwise, I receive emails and messages on how all my (free) material on physics, math, and engineering have 
 * helped students with their studies, and I know what it's like to not have money as a student, but love physics 
 * (or math, sciences, etc.), so I am committed to keeping all my material open-source and free, whether or not 
 * sufficiently crowdfunded, under the open-source MIT license: 
 *  feel free to copy, edit, paste, make your own versions, share, use as you wish.    
 * Peace out, never give up! -EY
 * 
 * */
void deleteBeforeThrow(int i)
{
  int* p = new int[10];
  if (i < 0)
  {
    delete[] p; // delete before the throw or leak
    throw i;
  }
}

int main()
{
//  deleteBeforeThrow(4);  // this WORKS

  deleteBeforeThrow(-5);  // terminate called after throwing an instance of 'int'
  // Aborted (core dumped)

  return 0;
}