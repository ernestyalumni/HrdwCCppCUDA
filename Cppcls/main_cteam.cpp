/**
 * 	@file 	main_cteam.h
 * 	@brief 	apply PIMPL idiom to class CTeam. Advantage of pimpl: header files don't contain any implementation details.    
 * 	@ref	http://www.embeddeduse.com/2016/05/30/best-friends-cpp11-move-semantics-and-pimpl/
 * 	@details public interface, public, of CTeam is same as before.  
 * Replace private data members by a unique_ptr, and moved them into private implementation class CTeam::Impl 
 * Declaration and definition of CTeam::Impl located in source file cteam.cpp  
 * 1 of the biggest advantages of pimpl: header files don't contain any implementation details.    
 * 
 * COMPILATION TIP : -Wall warning all, -g debugger, gdb can step through 
 * g++ -Wall -g cteam.cpp main_cteam.cpp -o main_cteam
 * */
#include "cteam.h" // CTeam
#include <iostream> 

int main(int argc, char* argv[]) {
	CTeam cteam_objinst();
	CTeam cteam_objinst1(" cteam_objinst1 string ", 12, 13); 
	CTeam * cteam_ptr = new CTeam(); 
	CTeam * cteam_ptr1 = new CTeam(" cteam_ptr1 string ", 14,15);

/*
	std::cout << " cteam_objinst : " << cteam_objinst.name() << " " << cteam_objinst.points() << 
		" " << cteam_objinst.goalDifference() << std::endl; 
*/

	std::cout << std::endl << " cteam_objinst1 : " << cteam_objinst1.name() << " " << 
		cteam_objinst1.points() << " " << cteam_objinst1.goalDifference() << std::endl ; 

	std::cout << std::endl << " cteam_ptr1 : " << cteam_ptr1->name() << " " << 
		cteam_ptr1->points() << " " << cteam_ptr1->goalDifference() << std::endl ; 


	delete cteam_ptr;
	delete cteam_ptr1;
}
