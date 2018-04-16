/**
 * @file   : FilesAndfopen.cpp
 * @author : Ernest Yeung
 * @email  : ernestyalumni@gmail.com
 * @brief  : Demonstrate FILE type and fopen 
 * @ref    : http://en.cppreference.com/w/c/io/fopen
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
 * COMPILATION TIPS:
 *  g++ -std=c++17 -c factor.cpp
 * */
#include <stdio.h> // FILE, fopen, perror
#include <stdlib.h> // EXIT_FAILURE

int main(void)
{
  FILE* fp = fopen("test.txt", "r");
  if (!fp)
  {
    perror("File opening failed");
    return EXIT_FAILURE;
  }

  int c;  // note: int, not char, required to handle EOF
  while ((c = fgetc(fp)) != EOF) // standard C I/O file reading loop
  {
    putchar(c);
  }

  if (ferror(fp))
  {
    puts("I/O error when reading");
  }
  else if (feof(fp))
  {
    puts("End of file reached successfully");
  }

  fclose(fp);
}