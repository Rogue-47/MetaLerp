#ifndef PLATFORM_DEFS_H
#define PLATFORM_DEFS_H
//platform/toolchain-pecific definitions header for metalerp - deals with cross-OS/compiler definitions

/*OS-specific*/
    #if _WIN32

        #define WIN32_LEAN_AND_MEAN
        #include <Windows.h>
        #define GET_FREQ(freq) QueryPerformanceFrequency((freq))
        #define MEASURE(start_or_end) QueryPerformanceCounter((start_or_end))
        #define calcElapsed(frequency, start, end) ( ( (double) ( ((end).QuadPart) - ((start).QuadPart) ) ) / ( (double) ((frequency).QuadPart) ) )
        #define SLEEP(ms) Sleep((ms))


        #define METALERP_HEURISTICS_VARIABLES LARGE_INTEGER FREQUENCY, START, END;

        //compiler-specifics
        #ifdef _MSC_VER
            #define METALERP_MACRO_PRAGMA(arg) __pragma(#arg)

        #endif

    #else /*assumed posix for now*/

        #include <unistd.h>
        #define GET_FREQ(freq)
        #define MEASURE(start_or_end) clock_gettime(CLOCK_MONOTONIC, (start_or_end))
        #define calcElapsed(frequency, start, end) ((end).tv_nsec) < ((start).tv_nsec) ? \
                (( (double)( ((end).tv_sec) - ((start).tv_sec)) ) - 1.0) + ( ((double) ( ( (((end).tv_nsec) + 1000000000L) - ((start).tv_nsec)) ) / (1e+9)) )   \
            :   ( (double)( ((end).tv_sec) - ((start).tv_sec)) ) + ( ((double)( ((end).tv_nsec) - ((start).tv_nsec) ) ) / (1e+9))
        
        #define SLEEP(ms) usleep(((unsigned int)(ms))*((unsigned int)1000)) /*this takes microsecs so the conversion is necessary*/


        #define METALERP_HEURISTICS_VARIABLES struct timespec START, END;
    
        //compiler-specifics
        #define METALERP_MACRO_PRAGMA(arg) _Pragma(#arg)


    #endif

#endif //platform definitions header