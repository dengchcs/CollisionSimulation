#ifndef HELPERS_CUH
#define HELPERS_CUH

/**
 * @brief 检查cuda执行结果, 在出错时报错并终止程序.
 */
void check(cudaError_t error, const char *name);

#endif