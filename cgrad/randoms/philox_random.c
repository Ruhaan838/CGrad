#include <stdio.h>
#include <stdint.h>
#include <time.h>

#define PHILOX_ROUNDS 10
#define PHILOX_M0 0xD2511F53
#define PHILOX_M1 0xCD9E8D57

void philox4x32(int *ctr, int *key) {
    for (int i = 0; i < PHILOX_ROUNDS; i++) {
        int hi0 = ((uint64_t)PHILOX_M0 * ctr[0]) >> 32;
        int lo0 = PHILOX_M0 * ctr[0];
        int hi1 = ((uint64_t)PHILOX_M1 * ctr[2]) >> 32;
        int lo1 = PHILOX_M1 * ctr[2];

        ctr[0] = hi1 ^ ctr[1] ^ key[0];
        ctr[1] = lo1;
        ctr[2] = hi0 ^ ctr[3] ^ key[1];
        ctr[3] = lo0;

        key[0] += 0x9E3779B9; 
        key[1] += 0xBB67AE85;
    }
}

float philox4x32_float_n(int seed, int counter) {
    int ctr[4] = {counter, 0, 0, 0};
    int key[2] = {seed, seed ^ 0xDEADBEEF}; 

    philox4x32(ctr, key);

    int result = ctr[0] ^ ctr[2];
    float ans = result / (float)UINT32_MAX;
    float f_ans = ans < 0 ? ans * -1. : ans;
    return f_ans * 2;
}

float philox4x32_float(int seed, int counter, float min, float max) {
    int ctr[4] = {counter, 0, 0, 0};
    int key[2] = {seed, seed ^ 0xDEADBEEF}; 

    philox4x32(ctr, key);

    int result = ctr[0] ^ ctr[2];
    float ans = result / (float)UINT32_MAX; 
    float f_ans = ans < 0 ? ans * -1. : ans;
    return (min + f_ans * (max - min)) * 2;
}
