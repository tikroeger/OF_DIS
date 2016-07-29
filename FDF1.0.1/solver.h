#include <stdio.h>
#include <stdlib.h>

#include "image.h"

#ifdef __cplusplus
extern "C" {
#endif

// Perform n iterations of the sor_coupled algorithm for a system of the form as described in opticalflow.c
void sor_coupled(image_t *du, image_t *dv, image_t *a11, image_t *a12, image_t *a22, const image_t *b1, const image_t *b2, const image_t *dpsis_horiz, const image_t *dpsis_vert, const int iterations, const float omega);

void sor_coupled_slow_but_readable(image_t *du, image_t *dv, image_t *a11, image_t *a12, image_t *a22, const image_t *b1, const image_t *b2, const image_t *dpsis_horiz, const image_t *dpsis_vert, const int iterations, const float omega);

void sor_coupled_slow_but_readable_DE(image_t *du, const image_t *a11, const image_t *b1, const image_t *dpsis_horiz, const image_t *dpsis_vert, const int iterations, const float omega);

#ifdef __cplusplus
}
#endif
