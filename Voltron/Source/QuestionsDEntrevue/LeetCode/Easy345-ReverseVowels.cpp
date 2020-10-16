//------------------------------------------------------------------------------
/// \file LeetCodePlayground.cpp
/// \author
/// \brief .
/// \ref 
/// 2020/10/14 01:21 Start.
/// 2020/10/14 03:49 Redid Python implementation.
///-----------------------------------------------------------------------------

// https://leetcode.com/problems/reorder-data-in-log-files/
#include <algorithm> // std::swap
#include <cstddef> // std::size_t
#include <iostream>
#include <iterator> // std::distance
#include <string>
#include <unordered_set>
#include <vector>

using std::distance;
//using std::stack;
using std::size_t;
using std::string;
using std::swap;
using std::vector;

string example_input_1 {"hello"};
string example_input_2 {"leetcode"};
string example_input_3 {"aA"};

const string example_output_1 {"holle"};
const string example_output_2 {"leotcede"};

const std::unordered_set<char> vowels {
  'a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U'};

/* Wrong
vector<std::size_t> get_vowel_indices(const string& s)
{
  vector<std::size_t> vowel_indices;

  for (auto iter {s.begin()}; iter != s.end(); ++iter)
  {
    // Found letter in string to be a vowel.
    if (vowels.find(*iter) != vowels.end())
    {
      vowel_indices.emplace_back(distance(s.begin(), iter));
    }
  } 

  return vowel_indices;
}

swap_vowels(string& s, vector<size_t> vowel_indices)
*/

/*
bool is_vowel(char& x)
{
  // Found x in vowels.
  if (vowels.find(x) != vowels.end())
  {
    return true;
  }
  return false;
}
*/

bool is_vowel(char &c)
{
  switch(c)
  {
    case 'a':
    case 'e':
    case 'i':
    case 'o':
    case 'u':
    case 'A':
    case 'E':
    case 'I':
    case 'O':
    case 'U':
        return true;
    default: return false;
  }
}

string reverse_vowels_from_ends(string& s)
{
  /*
  const size_t N {s.size()};
  size_t l {0};
  size_t r {N - 1};
  */
  int l = 0;
  int r = s.size() - 1;

  while (l < r)
  {
    if (is_vowel(s[l]) && is_vowel(s[r]))
    {
      swap(s[l], s[r]);

      ++l;
      --r;
    }

    if (!is_vowel(s[l]))
    {
      ++l;
    }

    if (!is_vowel(s[r]))
    {
      --r;
    }
  }
  return s;
}

class Solution
{
  public:

    string reverseVowels(string s)
    {
      return s;
    }
};

int main()
{ 
  std::cout << reverse_vowels_from_ends(example_input_1) << "\n";
  std::cout << reverse_vowels_from_ends(example_input_2) << "\n";
  std::cout << reverse_vowels_from_ends(example_input_3) << "\n";

}


// cf. https://leetcode.com/problems/reverse-vowels-of-a-string/discuss/891686/C%2B%2B-8ms-greater-~97
// C++ , 8ms , > ~97%
/*

class Solution {
public:
    
    bool isvowel(char &c){
        switch(c){
            case 'a':
            case 'e':
            case 'i':
            case 'o':
            case 'u':
            case 'A':
            case 'E':
            case 'I':
            case 'O':
            case 'U':
                return true;
            default: return false;
        }
    }
    
    
    string reverseVowels(string s) {
        
        int i=0, j=s.size()-1;
        
        if(j+1==1) return s;
        
        while(i<j){
            
            if(isvowel(s[i]) && isvowel(s[j])){
                swap(s[i],s[j]);
                ++i;
                --j;
                
            } else {
                
                if(!isvowel(s[i])) {
                    i++;
                }
            
                if(!isvowel(s[j])) {
                    j--;
                }
                
            }
            
        }
        return s;
    }
};

*/