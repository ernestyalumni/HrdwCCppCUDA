#include <algorithm>
#include <cstdio>
#include <iostream>

int solve_me_first(int a, int b)
{
  return a + b;
}

int main()
{
  // Solve Me First
  // cf. https://www.hackerrank.com/challenges/solve-me-first/problem
  {
    std::cout << "\n Solve Me First begins \n";
    int num1, num2;
    int sum;

    // std::cin >> num1 >> num2;
    num1 = 2;
    num2 = 3;
    sum = solve_me_first(num1, num2);
    std::cout << sum;
    std::cout << "\n Solve Me First ends \n";

  }

  // https://www.hackerrank.com/challenges/cpp-hello-world/problem
  {
    std::cout << "Hello, World!";
  }

  // https://www.hackerrank.com/challenges/cpp-input-and-output/problem
  {
    std::cout << "\n Input and Output begins \n";

    int a, b, c;

    // std::cin >> a >> b >> c;
    std::cout << (a + b + c) << std::endl;

    std::cout << "\n Input and Output end \n";
  }

  // https://www.hackerrank.com/challenges/c-tutorial-basic-data-types/problem  
  {
    std::cout << "\n Basic Data Types begins \n";

    int x_i; // %d 32-bit int
    long x_l; // %ld 64-bit int
    char x_c; // %c character type
    float x_f; // %f 32-bit float
    double x_d; // %lf 64-bit real 

    //scanf("%d %ld %c %f %lf", &x_i, &x_l, &x_c, &x_f, &x_d);

    x_i = 3;
    x_l = 12345678912345;
    x_c = 'a';
    x_f = 334.23;
    x_d = 14049.30493;

    printf("%d\n%ld\n%c\n%.3f\n%.10lf\n", x_i, x_l, x_c, x_f, x_d);

    std::cout << "\n Basic Data Types ends \n";
  }

  // 


  return 0;
}