/**
 * @file   : functiontemplate.h
 * @author : Ernest Yeung
 * @email  : ernestyalumni@gmail.com
 * @brief  : Program to demonstrate function templates.
 * @ref    : pp. 684 23.5 Function Templates, Ch. 23 Templates; Bjarne Stroustrup, The C++ Programming Language, 4th Ed., Stroustrup  
 * https://www.geeksforgeeks.org/templates-cpp/
 * https://www.geeksforgeeks.org/template-specialization-c/
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

 /**
  * @brief One function works for all data types.  This would work 
  * even for user defined types if operator '>' is overloaded
  */ 
#include <iostream>

#include <vector>

template <typename T>
T myMax(T x, T y)
{
    return (x > y) ? x : y;
};

/** @ref https://www.geeksforgeeks.org/template-specialization-c/ 
 * @brief function template specialization - we have general template fun() for 
 * all data types except int.  For int, there's a specialized version of fun().
 * */
template <class T>
void fun(T a)
{
    std::cout << "The main template fun(): "
                << a << std::endl;
};

template<>
void fun(int a)
{
    std::cout << "Specialized Template for int type: " 
                << a << std::endl; 
};

/**
 * @ref pp. 684 23.5 Function Templates, Ch. 23 Templates; Bjarne Stroustrup, The C++ Programming Language, 4th Ed., Stroustrup
 * @brief Shell sort (Knuth, Vol. 3. pg. 84)
 * */

template<typename T>
void sort(std::vector<T>& v)        // definition
{
    const size_t n = v.size();

    for (int gap = n/2; 0 < gap; gap /= 2)
    {
        for (int i = gap; i < n; i++)
        {
            for (int j = i - gap; 0 <= j; j-=gap)
            {
                if (v[j+gap] < v[j]) 
                {   // swap v[j] and v[j + gap]
                    T temp = v[j];
                    v[j] = v[j + gap];
                    v[j + gap] = temp;
                }
            }
        }
    }
};

template<typename T>
void sort_swap(std::vector<T>& v)       // definition
{
    const size_t n = v.size();

    for (int gap = n/2; 0 < gap; gap /= 2)
    {
        for (int i = gap; i < n; i++)
        {
            for (int j = i - gap; 0 <= j; j-=gap)
            {
                if (v[j+gap] < v[j])
                {   // swap v[j] and v[j + gap]
                    std::swap(v[j], v[j + gap]);
                }
            }
        }
    }
};

template<typename T>
void sort_explicit(std::vector<T>& v)       // definition
{
    const size_t n = v.size();
    std::cout << std::endl << " n : " << n << std::endl;
    for (int gap = n/2; 0 < gap; gap /= 2)
    {
        std::cout << " gap : " << gap << std::endl;
        for (int i = gap; i < n; i ++)
        {
            std::cout << " i : " << std::endl; 
            for (int j = i - gap; 0 <= j; j-=gap)
            {
                std::cout << " j : " << j << " ";
                if (v[j+gap] < v[j])
                {   // swap v[j] and v[j + gap]
                    std::swap(v[j], v[j + gap]);
                }
            }
        }
    }
};


