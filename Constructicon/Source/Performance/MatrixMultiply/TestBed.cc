#include "MatrixMultiply.h"

#include <iostream>
#include <unistd.h>

using Performance::MatrixMultiply::Mat;
using std::cout;

int main(int argc, char** argv)
{
  int optchar {0};
  int show_usec {0};
  int should_print {0};
  int use_zero_matrix {0};

  while ((optchar = getopt(argc, argv, "upz")) != -1)
  {
    switch (optchar)
    {
      case 'u':
        show_usec = 1;
        break;
      case 'p':
        should_print = 1;
        break;
      case 'z':
        use_zero_matrix = 1;
        break;
      default:
        cout << "Ignoring unrecognized option: " << static_cast<char>(optchar)
          << "\n";
        continue;
    }
  }

  cout << show_usec << should_print << use_zero_matrix << "\n";
}