/** ****************************************************************************
 *  @file    trace.hpp
 *  @brief   Debug macros definition
 *  @author  Roberto Valle Fernandez
 *  @date    2012/01
 *  @copyright All rights reserved.
 *  Software developed by UPM PCR Group: http://www.dia.fi.upm.es/~pcr
 ******************************************************************************/

// ------------------ RECURSION PROTECTION -------------------------------------
#ifndef TRACE_HPP
#define TRACE_HPP

// ----------------------- INCLUDES --------------------------------------------
#include <iostream>

#define PRINT(...) std::cout << __VA_ARGS__ << std::endl;
#define ERROR(...) std::cerr << __VA_ARGS__ << std::endl;

#ifdef _DEBUG
  #define TRACE(...) std::cout << __VA_ARGS__ << std::endl;
  #define TRACE_INFO(...) std::cout << __FILE__ << "(" << __LINE__ << "):" << __VA_ARGS__ << std::endl;
#else
  #define TRACE(...)
  #define TRACE_INFO(...)
#endif

#endif /* TRACE_HPP */
