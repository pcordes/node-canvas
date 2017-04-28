#ifndef PIXEL_FORMAT_H
#define PIXEL_FORMAT_H

#include <stdint.h>
#include <stddef.h> // TODO: <cstddef>?

#ifdef _MSC_VER
#define __restrict__ __restrict
#elif defined(__GNUC__)
// __restrict__ works like C11's restrict keyword
#else
#define __restrict__   // no-op
#endif


// Output and input buffers must not overlap.

// big-endian RGBA -> alpha-premultiplied native-endian ARGB
void PutPixels(uint32_t *__restrict__ dst_native, const uint32_t *__restrict__ src_RGBA,
	       size_t rows, size_t cols, size_t dstPixStride, size_t srcPixStride);

// alpha-premultiplied native-endian ARGB -> big-endian RGBA (de-multiply alpha)
void GetPixels(uint32_t *__restrict__ dst_RGBA, const uint32_t *__restrict__ src_native,
	       size_t rows, size_t cols, size_t dstPixStride, size_t srcPixStride);

#endif // PIXEL_FORMAT_H
