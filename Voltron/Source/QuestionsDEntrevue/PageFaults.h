#ifndef QUESTIONS_D_ENTREVUE_PAGE_FAULTS_H
#define QUESTIONS_D_ENTREVUE_PAGE_FAULTS_H

#include <algorithm>
#include <vector>

namespace QuestionsDEntrevue
{

//------------------------------------------------------------------------------
/// cf. https://www.geeksforgeeks.org/page-faults-in-lru-implementation/
/// \brief Counts number of page faults.
/// \details LRU - Least Recently Used (LRU) Algorithm.
/// When a new page is referred to and isn't present in memory, the page fault
/// occurs and the OS replaces one of the existing pages with a newly needed
/// page.
/// LRU is one such page replacement policy in which the least recently used
/// pages are replaced.
//------------------------------------------------------------------------------
int page_faults(const int n, const int c, int pages[]);

} // namespace QuestionsDEntrevue

#endif // QUESTIONS_D_ENTREVUE_PAGE_FAULTS_H