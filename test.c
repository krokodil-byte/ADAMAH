/*
 * ADAMAH Test Suite - Maps + Math
 */
#include "adamah.h"
#include <stdio.h>
#include <math.h>

#define TEST(cond, msg) do { if (!(cond)) { printf("FAIL: %s\n", msg); return 1; } } while(0)
#define NEAR(a, b) (fabsf((a) - (b)) < 1e-4)

int main(void) {
    printf("=== ADAMAH Unified Test Suite ===\n\n");
    
    TEST(adamah_init() == ADAMAH_OK, "init");
    printf("✓ adamah_init\n");
    
    // === MAP TESTS ===
    printf("\n--- Map Tests ---\n");
    
    TEST(map_init(0, 32, 256, 256) == ADAMAH_OK, "map_init");
    printf("✓ map_init (word_size=32, 64K elements)\n");
    
    uint64_t locs[3] = {100, 200, 300};
    float vals[24] = {1,2,3,4,5,6,7,8, 9,10,11,12,13,14,15,16, 17,18,19,20,21,22,23,24};
    TEST(mscatter(0, locs, vals, 3) == ADAMAH_OK, "mscatter");
    printf("✓ mscatter 3 elements\n");
    
    float out[24] = {0};
    TEST(mgather(0, locs, out, 3) == ADAMAH_OK, "mgather");
    TEST(NEAR(out[0], 1) && NEAR(out[8], 9) && NEAR(out[16], 17), "gather values");
    printf("✓ mgather verified\n");
    
    TEST(map_save(0, "/tmp/test_map.bin") == ADAMAH_OK, "save");
    map_clear(0);
    TEST(map_load(0, "/tmp/test_map.bin") == ADAMAH_OK, "load");
    mgather(0, locs, out, 3);
    TEST(NEAR(out[0], 1), "load verify");
    printf("✓ map_save/load\n");
    
    map_destroy(0);
    
    // === VBUF TESTS ===
    printf("\n--- VBuf Tests ---\n");
    
    TEST(vbuf_alloc(0, 16) == ADAMAH_OK, "vbuf_alloc 0");
    TEST(vbuf_alloc(1, 16) == ADAMAH_OK, "vbuf_alloc 1");
    TEST(vbuf_alloc(2, 16) == ADAMAH_OK, "vbuf_alloc 2");
    printf("✓ vbuf_alloc\n");
    
    float a[] = {1, 2, 3, 4, 5, 6, 7, 8};
    float b[] = {8, 7, 6, 5, 4, 3, 2, 1};
    vbuf_upload(0, a, 0, 8);
    vbuf_upload(1, b, 0, 8);
    printf("✓ vbuf_upload\n");
    
    // === MATH TESTS ===
    printf("\n--- Math Tests ---\n");
    
    vop2(VOP_ADD, 2, 0, 1, 8);
    float r[8];
    vbuf_download(2, r, 0, 8);
    TEST(NEAR(r[0], 9) && NEAR(r[7], 9), "ADD");
    printf("✓ vop2 ADD: 1+8=%.0f, 8+1=%.0f\n", r[0], r[7]);
    
    vop2(VOP_MUL, 2, 0, 1, 8);
    vbuf_download(2, r, 0, 8);
    TEST(NEAR(r[0], 8) && NEAR(r[3], 20), "MUL");
    printf("✓ vop2 MUL: 1*8=%.0f, 4*5=%.0f\n", r[0], r[3]);
    
    vop_scalar(VOP_MUL, 2, 0, 10.0f, 8);
    vbuf_download(2, r, 0, 8);
    TEST(NEAR(r[0], 10) && NEAR(r[4], 50), "scalar MUL");
    printf("✓ vop_scalar: 1*10=%.0f, 5*10=%.0f\n", r[0], r[4]);
    
    float t[] = {0, 1, -1, 2, -2, 0, 0, 0};
    vbuf_upload(0, t, 0, 8);
    vop1(VOP_RELU, 2, 0, 5);
    vbuf_download(2, r, 0, 5);
    TEST(NEAR(r[0], 0) && NEAR(r[1], 1) && NEAR(r[2], 0), "RELU");
    printf("✓ vop1 RELU: relu(-1)=%.0f, relu(1)=%.0f\n", r[2], r[1]);
    
    float e[] = {0, 1, 0, 0, 0, 0, 0, 0};
    vbuf_upload(0, e, 0, 8);
    vop1(VOP_EXP, 2, 0, 2);
    vbuf_download(2, r, 0, 2);
    TEST(NEAR(r[0], 1.0) && NEAR(r[1], 2.7183), "EXP");
    printf("✓ vop1 EXP: e^0=%.2f, e^1=%.2f\n", r[0], r[1]);
    
    float s[] = {1, 2, 3, 4, 5, 0, 0, 0};
    vbuf_upload(0, s, 0, 8);
    vreduce(VRED_SUM, 2, 0, 5);
    vbuf_download(2, r, 0, 1);
    TEST(NEAR(r[0], 15), "SUM");
    printf("✓ vreduce SUM: 1+2+3+4+5=%.0f\n", r[0]);
    
    float da[] = {1, 2, 3, 4};
    float db[] = {4, 3, 2, 1};
    vbuf_upload(0, da, 0, 4);
    vbuf_upload(1, db, 0, 4);
    vdot(2, 0, 1, 4);
    vbuf_download(2, r, 0, 1);
    TEST(NEAR(r[0], 20), "DOT");
    printf("✓ vdot: [1,2,3,4]·[4,3,2,1]=%.0f\n", r[0]);
    
    float sm[] = {1, 2, 3, 4};
    vbuf_upload(0, sm, 0, 4);
    vsoftmax(0, 0, 4);
    vbuf_download(0, r, 0, 4);
    float sum = r[0]+r[1]+r[2]+r[3];
    TEST(NEAR(sum, 1.0), "softmax sum");
    printf("✓ vsoftmax: sum=%.4f\n", sum);
    
    vbuf_free(0); vbuf_free(1); vbuf_free(2);
    adamah_shutdown();
    
    printf("\n=== ALL TESTS PASSED ===\n");
    return 0;
}
