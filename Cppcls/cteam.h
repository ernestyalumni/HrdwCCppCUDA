/**
 * 	@file 	cteam.h
 * 	@brief 	apply PIMPL idiom to class CTeam. Advantage of pimpl: header files don't contain any implementation details.    
 * 	@ref	http://www.embeddeduse.com/2016/05/30/best-friends-cpp11-move-semantics-and-pimpl/
 * 	@details public interface, public, of CTeam is same as before.  
 * Replace private data members by a unique_ptr, and moved them into private implementation class CTeam::Impl 
 * Declaration and definition of CTeam::Impl located in source file cteam.cpp  
 * 1 of the biggest advantages of pimpl: header files don't contain any implementation details.    
 * 
 * COMPILATION TIP : -Wall warning all, -g debugger, gdb can step through 
 * g++ -Wall -g virtualfunc.cpp -o virtualfunc
 * */

#ifndef CTEAM_H
#define CTEAM_H

#include <memory>  

class CTeam
{
	public:
		// destructor
		~CTeam();
		// default constructor
		CTeam();	
		// name constructor
		CTeam(const std::string &n, int p, int gd);
		
		// copy constructor
		CTeam(const CTeam &t);
		// copy assignment
		CTeam &operator=(const CTeam &t);  
		
		// move constructor 
		CTeam(CTeam &&t); 
		// move assignment 
		CTeam &operator=(CTeam &&t); 
		
		std::string name() const;
		int points() const;
		int goalDifference() const; 
		
	private:
		struct Impl;

		// 
		std::unique_ptr<Impl> m_impl;
};

#endif // CTEAM_H
