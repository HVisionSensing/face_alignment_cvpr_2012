/** ****************************************************************************
 *  @file    Timing.hpp
 *  @brief   Real-time facial feature detection
 *  @author  Matthias Dantone
 *  @date    2011/05
 ******************************************************************************/

// ------------------ RECURSION PROTECTION -------------------------------------
#ifndef TIMING_HPP
#define TIMING_HPP

// ----------------------- INCLUDES --------------------------------------------
#include <trace.hpp>
#include <sys/time.h>

/** ****************************************************************************
 * @class Timing
 * @brief Time in milliseconds
 ******************************************************************************/
class Timing
{
public:
  Timing()
  {
    start();
  };

  void
  start()
  {
    gettimeofday(&m_time, NULL);
  };

  float
  restart()
  {
    float val = elapsed();
    gettimeofday(&m_time, NULL);
    return val;
  };

  float
  elapsed()
  {
    return now() - (*this);
  };

  void
  print
    (
    const char *name
    )
  {
    float time = elapsed();
    PRINT(name << ": took " << time << " milliseconds");
    restart();
  };

  static inline Timing
  now()
  {
    Timing t;
    t.start();
    return t;
  };

  float
  operator -
    (
    const Timing &t1
    )
  {
    return (float) 1000.0f * (m_time.tv_sec - t1.m_time.tv_sec) + 1.0e-3f * (m_time.tv_usec - t1.m_time.tv_usec);
  };

private:
  timeval m_time;
};

#endif /* TIMING_HPP */
