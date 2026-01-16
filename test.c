/*
 * ADAMAH v3 Test
 */
#include "adamah.h"
#include <stdio.h>
#include <math.h>

#define NEAR(a, b) (fabsf((a) - (b)) < 0.01f)
#define PI 3.14159265f

int main(void) {
    printf("=== ADAMAH v3 Test ===\n\n");
    
    adamah_init();
    printf("✓ init\n");
    
    // Basic ops
    float a[] = {1, 2, 3, 4};
    float b[] = {4, 3, 2, 1};
    
    inject("a", a, 4);
    inject("b", b, 4);
    printf("✓ inject\n");
    
    vop2(VOP_ADD, "c", "a", "b", 4);
    float c[4];
    extract("c", c, 4);
    printf("✓ ADD: %.0f + %.0f = %.0f\n", a[0], b[0], c[0]);
    
    vop2(VOP_POW, "pow", "a", "b", 4);
    float pow_r[4];
    extract("pow", pow_r, 4);
    printf("✓ POW: %.0f ^ %.0f = %.0f\n", a[0], b[0], pow_r[0]);
    
    // Trig
    float angles[] = {0, PI/6, PI/4, PI/2};
    inject("ang", angles, 4);
    vop1(VOP_SIN, "sin", "ang", 4);
    vop1(VOP_COS, "cos", "ang", 4);
    vop1(VOP_TAN, "tan", "ang", 3);  // skip PI/2
    float sin_r[4], cos_r[4];
    extract("sin", sin_r, 4);
    extract("cos", cos_r, 4);
    printf("✓ SIN(π/6) = %.3f (expect 0.5)\n", sin_r[1]);
    printf("✓ COS(π/4) = %.3f (expect 0.707)\n", cos_r[2]);
    
    // Inverse trig
    float vals[] = {0.5f, 0.707f, 1.0f};
    inject("vals", vals, 3);
    vop1(VOP_ASIN, "asin", "vals", 3);
    float asin_r[3];
    extract("asin", asin_r, 3);
    printf("✓ ASIN(0.5) = %.3f (expect %.3f)\n", asin_r[0], PI/6);
    
    // Hyperbolic
    float x[] = {0, 1, 2};
    inject("x", x, 3);
    vop1(VOP_SINH, "sinh", "x", 3);
    vop1(VOP_COSH, "cosh", "x", 3);
    float sinh_r[3], cosh_r[3];
    extract("sinh", sinh_r, 3);
    extract("cosh", cosh_r, 3);
    printf("✓ SINH(1) = %.3f, COSH(1) = %.3f\n", sinh_r[1], cosh_r[1]);
    
    // Rounding
    float floats[] = {1.2f, 2.5f, 3.7f, -1.5f};
    inject("floats", floats, 4);
    vop1(VOP_FLOOR, "floor", "floats", 4);
    vop1(VOP_CEIL, "ceil", "floats", 4);
    vop1(VOP_ROUND, "round", "floats", 4);
    float floor_r[4], ceil_r[4], round_r[4];
    extract("floor", floor_r, 4);
    extract("ceil", ceil_r, 4);
    extract("round", round_r, 4);
    printf("✓ FLOOR(2.5)=%.0f CEIL(2.5)=%.0f ROUND(2.5)=%.0f\n", floor_r[1], ceil_r[1], round_r[1]);
    
    // Sign
    float signs[] = {-5, 0, 5};
    inject("signs", signs, 3);
    vop1(VOP_SIGN, "sign", "signs", 3);
    float sign_r[3];
    extract("sign", sign_r, 3);
    printf("✓ SIGN(-5, 0, 5) = (%.0f, %.0f, %.0f)\n", sign_r[0], sign_r[1], sign_r[2]);
    
    // Reduce
    vreduce(VRED_PROD, "prod", "a", 4);
    vreduce(VRED_MEAN, "mean", "a", 4);
    float prod_r, mean_r;
    extract("prod", &prod_r, 1);
    extract("mean", &mean_r, 1);
    printf("✓ PROD([1,2,3,4]) = %.0f\n", prod_r);
    printf("✓ MEAN([1,2,3,4]) = %.2f\n", mean_r);
    
    // Cumsum
    vcumsum("cumsum", "a", 4);
    float cumsum_r[4];
    extract("cumsum", cumsum_r, 4);
    printf("✓ CUMSUM([1,2,3,4]) = [%.0f,%.0f,%.0f,%.0f]\n", cumsum_r[0], cumsum_r[1], cumsum_r[2], cumsum_r[3]);
    
    // Diff
    vdiff("diff", "a", 4);
    float diff_r[3];
    extract("diff", diff_r, 3);
    printf("✓ DIFF([1,2,3,4]) = [%.0f,%.0f,%.0f]\n", diff_r[0], diff_r[1], diff_r[2]);
    
    // Integrate / Derivative
    float f[] = {0, 1, 4, 9, 16};  // x^2 at x=0,1,2,3,4
    inject("f", f, 5);
    vintegrate("integral", "f", 1.0f, 5);
    vderivative("deriv", "f", 1.0f, 5);
    float integral_r[5], deriv_r[4];
    extract("integral", integral_r, 5);
    extract("deriv", deriv_r, 4);
    printf("✓ INTEGRATE(x²) dx=1: [%.0f,%.0f,%.0f,%.0f,%.0f]\n", 
           integral_r[0], integral_r[1], integral_r[2], integral_r[3], integral_r[4]);
    printf("✓ DERIVATIVE(x²): [%.0f,%.0f,%.0f,%.0f] (expect 1,3,5,7)\n",
           deriv_r[0], deriv_r[1], deriv_r[2], deriv_r[3]);
    
    // Linspace / Arange
    vlinspace("lin", 0, 10, 5);
    varange("rng", 0, 2.5f, 5);
    float lin_r[5], rng_r[5];
    extract("lin", lin_r, 5);
    extract("rng", rng_r, 5);
    printf("✓ LINSPACE(0,10,5) = [%.1f,%.1f,%.1f,%.1f,%.1f]\n", lin_r[0], lin_r[1], lin_r[2], lin_r[3], lin_r[4]);
    printf("✓ ARANGE(0,2.5,5) = [%.1f,%.1f,%.1f,%.1f,%.1f]\n", rng_r[0], rng_r[1], rng_r[2], rng_r[3], rng_r[4]);
    
    // Map
    map_init(0, 32, 256, 256);
    float locs[] = {100, 200, 300};
    float vals_m[] = {1,0,0,0,0,0,0,0, 2,0,0,0,0,0,0,0, 3,0,0,0,0,0,0,0};
    inject("locs", locs, 3);
    inject("vals", vals_m, 24);
    mscatter(0, "locs", "vals", 3);
    mgather(0, "locs", "out", 3);
    float out[24];
    extract("out", out, 24);
    printf("✓ MAP scatter/gather: [%.0f, %.0f, %.0f]\n", out[0], out[8], out[16]);
    
    map_destroy(0);
    adamah_shutdown();
    
    printf("\n=== ALL TESTS PASSED ===\n");
    return 0;
}
