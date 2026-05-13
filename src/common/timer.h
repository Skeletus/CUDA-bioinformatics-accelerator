#ifndef TIMER_H
#define TIMER_H

#include <chrono>

class CpuTimer {
public:
    CpuTimer() {
        reset();
    }

    void reset() {
        start_time_ = Clock::now();
    }

    double elapsed_milliseconds() const {
        const auto end_time = Clock::now();
        return std::chrono::duration<double, std::milli>(end_time - start_time_).count();
    }

private:
    using Clock = std::chrono::high_resolution_clock;
    Clock::time_point start_time_;
};

#endif  // TIMER_H
