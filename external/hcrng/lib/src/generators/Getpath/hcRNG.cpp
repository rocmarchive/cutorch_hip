
/* @file hcRNG.cpp
* @brief Implementation of functions defined in hcRNG.h
*/
#include "hcRNG/hcRNG.h"
#include <iostream>
#include <hc.hpp>
#include "hc_short_vector.hpp"

//using namespace hc;
using namespace hc;
using namespace hc::short_vector;
using namespace std;

#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <unistd.h>
#define CASE_ERR_(code,msg) case code: base = msg; break
#define CASE_ERR(code)      CASE_ERR_(HCRNG_ ## code, MSG_ ## code)


//tern char errorString[1024];
char errorString[1024]                          = "";
static const char MSG_DEFAULT[]                 = "unknown status";
static const char MSG_SUCCESS[]                 = "success";
static const char MSG_OUT_OF_RESOURCES[]        = "out of resources";
static const char MSG_INVALID_VALUE[]           = "invalid value";
static const char MSG_INVALID_RNG_TYPE[]        = "invalid type of RNG";
static const char MSG_INVALID_STREAM_CREATOR[]  = "invalid stream creator";
static const char MSG_INVALID_SEED[]            = "invalid seed";
static const char MSG_FUNCTION_NOT_IMPLEMENTED[]= "function not implemented";

const char* hcrngGetErrorString()
{
	return errorString;
}

static char lib_path_default[] = ".";

const char* hcrngGetLibraryRoot()
{
	const char* lib_path = getenv("HCRNG_PATH");

	if (lib_path == NULL) {
		return lib_path_default;
	}
	else
		return lib_path;
}

hcrngStatus hcrngSetErrorString(int err, const char* msg, ...)
{
    char formatted[1024];
    const char* base;
    switch (err) {
        CASE_ERR(SUCCESS);
        CASE_ERR(OUT_OF_RESOURCES);
        CASE_ERR(INVALID_VALUE);
        CASE_ERR(INVALID_RNG_TYPE);
        CASE_ERR(INVALID_STREAM_CREATOR);
        CASE_ERR(INVALID_SEED);
        CASE_ERR(FUNCTION_NOT_IMPLEMENTED);
        default: base = MSG_DEFAULT;
    }
    va_list args;
    va_start(args, msg);
    vsprintf(formatted, msg, args);
    sprintf(errorString, "[%s] %s", base, formatted);
    printf("%s\n", errorString);
    va_end(args);
        return (hcrngStatus)err;
}
