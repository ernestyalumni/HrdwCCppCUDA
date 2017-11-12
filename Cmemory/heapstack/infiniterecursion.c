/**
 * 	@file 	infiniterecursion.c
 * 	@brief 	stack overflow examples; infinite recursion 
 * 	@ref	http://www.inf.udec.cl/~leo/teoX.pdf
 * 	@details we blow stack by using more memory than is available to this thread.  
 * 	Notice also no base case for recursion 
 * 
 * COMPILATION TIP : -Wall warning all, -g debugger, gdb can step through 
 * gcc -Wall -g infiniterecursion.c -o infiniterecursion 
 * */
int add(int n)
{
	return n + add(n+1);
}  

int main(){
	add(2);
}
