//------------------------------------------------------------------------------
/// \file CStyleFileIO_main.cpp
//------------------------------------------------------------------------------

#include <cstdio> // std::FILE, std::fopen, std::remove
#include <cstdlib> // std::perror
#include <string>

using std::FILE; 
using std::fopen;
using std::remove;
using std::string;

int main()
{
  {
    // <cstdio>
    // std::FILE* fopen(const char* filename, const char* mode);

    const string filename {"test.txt"};
    const string current_directory {"./"};
    const string fullpath {current_directory + filename};

    //
    // If file is not there, failure to open and fclose would cause segmentation fault.
    //FILE* fp =  fopen(fullpath.data(), "r");
    // "w", write, Create a file for writing, destroy contents if file already
    // exists, create new file if it does not exist.
    FILE* fp =  fopen(fullpath.data(), "w");

    if (!fp)
    {
      std::perror("File opening failed");
    }


    fclose(fp);

    // Delete file.
    // int remove(const char* fname)
    const int remove_result {remove(fullpath.data())};
    
  }
  
}