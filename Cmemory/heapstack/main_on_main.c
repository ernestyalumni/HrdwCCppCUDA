/**
 * 	@file 	main_on_main.c
 * 	@brief 	stack overflow examples 
 * 	@ref	http://www.inf.udec.cl/~leo/teoX.pdf
 * 	@details we blow stack by using more memory than is available to this thread 
 * 
 * COMPILATION TIP : -Wall warning all, -g debugger, gdb can step through 
 * gcc -Wall -g main_on_main.c -o main_on_main
 * */
int main() 
{
	main();
}
