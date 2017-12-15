/**
 * 	@file 	cteam.cpp
 * 	@brief 	apply PIMPL idiom to class CTeam  
 * 	@ref	http://www.embeddeduse.com/2016/05/30/best-friends-cpp11-move-semantics-and-pimpl/
 * 	@details change implementation of pimpled class CTeam, without changing interface (in cteam.h)  
 * C++11 keyword default saves us from spelling out trivial implementation of destructor, 
 * copy constructor, and copy assignment operator of implementation class CTeam::Impl
 * We must only write the code for special name constructor 
 * COMPILATION TIP : -Wall warning all, -g debugger, gdb can step through 
 * g++ -Wall -g cteam.cpp -o cteam
 * */
#include "cteam.h"

#include <string> 
#include <vector>

struct CTeam::Impl  
{
	/* C++11 keyword default saves us from spelling out trivial implementation of destructor, 
	 * copy constructor, and copy assignment operator of Implementation class CTeam::Impl (struct?)
	 * */
	~Impl() = default; 
	Impl(const std::string &n, int p, int gd);
	Impl(const Impl &t) = default; 
	Impl &operator=(const Impl &t) = default; 
	
	std::string m_name; 
	int m_points;
	int m_goalDifference;
	static constexpr int statisticsSize = 100;
	std::vector<int> m_statistics; 
};

// constructor (body)
CTeam::Impl::Impl(const std::string &n, int p, int gd)
	: m_name(n) 
	, m_points(p) 
	, m_goalDifference(gd)
{
	m_statistics.reserve(statisticsSize);
	srand(p);
	for (int i = 0; i < statisticsSize; ++i) {
		m_statistics[i] = static_cast<int>(rand() % 10000) / 100.0;  
	}
}  

/*
 * We'll use CTeam::Impl to implement the constructors and assignment operators of the 
 * client-facing class CTeam. 
 * */

// let compiler generate destructor 
CTeam::~CTeam() = default;

CTeam::CTeam() : CTeam("", 0, 0) {}

// name constructor creates object CTeam::Impl with given arguments, as expected
CTeam::CTeam(const std::string &n, int p, int gd) 
	: m_impl(new Impl(n, p, gd))
{}

/* copy constructor and copy assignment must perform a deep copy.  
 * Compiler-generated versions would simply copy the unique pointer m_impl, that is, perform a shallow copy.  
 * As this is wrong, we must write code for copy constructor and assignment ourselves.  
 * */
CTeam::CTeam(const CTeam &t) 
	: m_impl(new Impl(*t.m_impl))
{}

CTeam &CTeam::operator=(const CTeam &t) 
{
	*m_impl = *t.m_impl;
	return *this;
}

// move constructor 
CTeam::CTeam(CTeam &&t) = default;

// move assignment 
CTeam &CTeam::operator=(CTeam &&t) = default;

// getting member functions  

std::string CTeam::name() const 
{
	return m_impl ? m_impl->m_name : "";
}

int CTeam::points() const 
{
	return m_impl ? m_impl->m_points : 0;
}

int CTeam::goalDifference() const 
{
	return m_impl ? m_impl->m_goalDifference : 0;
}
