/* Pixel format conversions: ImageData (RGBA big-endian) to/from cairo (ARGB native-endian)
 * applying or un-applying alpha pre-multiplication
 *
 * Authors: 2017 Peter Cordes <peter@cordes.ca>
 * Tuned to compile well with gcc for x86 or ARM (armv7-a).
 * Should be compiler-friendly in general, but I haven't checked.

 * clang does great with the x86 SIMD parts, but not well with the scalar.  So avoid clang for 32-bit x86 with SSE2 disabled.
 * I haven't looked at MSVC or icc.

 * Optimized for the alpha==0 || alpha==255 case.
 *  The full-complexity alpha case is necessarily much slower, but
 *  it's somewhat slower than it could be without the easy-alpha fast-path.

 * TODO: move the arch-specific SIMD code to per-arch .h files (so they can still inline)
 * TODO: runtime CPU detection?  SSSE3 on x86 makes a big difference, but AMD PhenomII CPUs don't have it and aren't *that* old
 * TODO: #include this whole file into CanvasRenderingContext2d.cc so function calls don't have to go through the PLT?
 *        Or so it can just inline, since the main put/get functions only have one caller

 * TODO: SIMD Context2d::blur, which also loops over pixels
 * TODO: SIMD for Image::decodeJPEGIntoSurface, RGB -> ARGB conversion setting alpha=255
 */

#include "pixformat.h"
#include "bitmanip.h"


#ifdef __SSE2__  // x86 SIMD
#include <immintrin.h>
#endif

// TODO: put this in a header, too?
#ifdef __GNUC__
// http://stackoverflow.com/questions/109710/likely-unlikely-macros-in-the-linux-kernel-how-do-they-work-whats-their
#define likely(x)       __builtin_expect(!!(x), 1)
#define unlikely(x)     __builtin_expect(!!(x), 0)
#else
#define likely(x)       (x)
#define unlikely(x)     (x)
#endif


#ifdef IACA_MARKS
//#include </opt/iaca-2.1/include/iacaMarks.h>
// works with or without -masm=intel, except with clang where it still breaks with -masm=intel :/
#define IACA_SSC_MARK( MARK_ID )                                        \
    __asm__ __volatile__ (                                              \
        "\n\t  {movl $"#MARK_ID", %%ebx|mov ebx,"#MARK_ID"}"            \
        "\n\t  .byte 0x64, 0x67, 0x90"                                  \
        : : : /* "memory" */ );
#define IACA_UD_BYTES __asm__ __volatile__ ("ud2");
#define IACA_START  IACA_UD_BYTES  IACA_SSC_MARK(111)
#define IACA_END IACA_SSC_MARK(222) IACA_UD_BYTES
#else
#define IACA_START
#define IACA_END
#endif



/********************** PutImage: shuffle and premultiply alpha ***************************/
#ifdef __SSE2__
#define HAVE_SIMD_PUTIMAGE 1
static inline __m128i flipRGBAtoNative(__m128i abgr)
{

#if defined( __SSSE3__) || defined(__AVX__)  // MSVC doesn't define __SSSE3__, only AVX if enabled
  // If this is inside the if(), gcc uses multiple copies of the shuffle constant when unrolling
  __m128i argb = _mm_shuffle_epi8(abgr, _mm_setr_epi8(2,1,0,3, 6,5,4,7, 10,9,8,11, 14,13,12,15));
#else
  __m128i word_swapped = _mm_shufflelo_epi16(abgr,         _MM_SHUFFLE(2,3, 0,1));
  word_swapped         = _mm_shufflehi_epi16(word_swapped, _MM_SHUFFLE(2,3, 0,1));
  // TODO: keep these split values around for the non-trivial alpha case?
  __m128i argb = _mm_or_si128(_mm_and_si128(abgr,         _mm_set1_epi16(0xff00)),   // keep G and A
			      _mm_and_si128(word_swapped, _mm_set1_epi16(0x00ff)) ); // keep R and B
  // andnot to share the mask would require an extra movdqa, because it destroys the inverted operand.
  // XOR, AND, XOR  could blend with only one mask constant, but less ILP.
#endif

  // alpha == 255 || alpha == 0 can be done with  unsigned(alpha - 1) > 253
  // but SSE2 only has signed comparisons, so we do  unsigned(alpha - 1) - 128 > (253-128)
  __m128i alpha_check = _mm_sub_epi8(abgr, _mm_set1_epi8(uint8_t(129)));
  alpha_check = _mm_cmpgt_epi8(alpha_check, _mm_set1_epi32((253U-128U)<<24) );  // 32-bit element size so we can use as a mask for the alpha==0 case?
  unsigned easy_alpha = _mm_movemask_ps(_mm_castsi128_ps(alpha_check));  // grab high bits of just the alpha pixels

  if (likely(easy_alpha == 0b1111)) { // all 4 elements have easy alpha
    // alpha==0 -> all-zero.  alpha=0xff -> unchanged
    __m128i alpha_signbit = _mm_srai_epi32(abgr, 31);  // broadcast just the sign bit.  24 would be the same since alpha=00 or ff
    __m128i argb_masked = _mm_and_si128(argb, alpha_signbit);
    return argb_masked;
  } else {   // Pre-multiply by alpha.
    // SSE2 can use bgra and rgba because pshufl/hw is a copy+shuffle
    // but for SSSE3 that slows down the fast case because pshufb is destructive (without AVX)

#if defined(__SSSE3__) || defined(__AVX__)
    //  http://wm.ite.pl/articles/sse4-alphaover.html.  ARGB->00A0A0A0 (A in the high half of words)
    __m128i lo = _mm_unpacklo_epi8(argb, _mm_setzero_si128());
    __m128i hi = _mm_unpackhi_epi8(argb, _mm_setzero_si128());
    __m128i alphamul_lo = _mm_shuffle_epi8(argb, _mm_setr_epi8(0xff, 0x3, 0xff, 0x3, 0xff, 0x3, 0xff, 0xff,
							       0xff, 0x7, 0xff, 0x7, 0xff, 0x7, 0xff, 0xff));
    __m128i alphamul_hi = _mm_shuffle_epi8(argb, _mm_setr_epi8(0xff, 11, 0xff, 11, 0xff, 11, 0xff, 0xff,
							       0xff, 15, 0xff, 15, 0xff, 15, 0xff, 0xff));
    lo = _mm_mulhi_epi16(lo, alphamul_lo);  // division by 256 happens for free with mulhuw and alpha byte position
    hi = _mm_mulhi_epi16(hi, alphamul_hi);
    __m128i premul_argb = _mm_packus_epi16(lo, hi);    // FIXME: divides by 256, not 255
#else // SSE2

    __m128i alpha = _mm_srli_epi32(argb, 24); // 000A

    __m128i ga = _mm_srli_epi16(argb, 8);  // 0A0G
    __m128i gmul = _mm_mullo_epi16(ga, alpha); // 00gg
              //    __m128i ga = _mm_and_si128(argb, _mm_set1_epi16(0xff00));  // and pmulhuw with 00A0?

    // multiply R and B together.  idea from http://stackoverflow.com/a/1103281/224132
              //    __m128i alpha2 = _mm_or_si128(alpha, _mm_slli_epi32(alpha, 16));  // 000A -> 0A0A   // PMULLD 0x10001 would work but is slow (and needs SSE4)
    __m128i packed_alpha = _mm_packs_epi32(alpha, alpha);  // 000W 000X ... -> 0W 0X 0Y 0Z | 0W 0X 0Y 0Z.  Can use a dead dst and then unpackhi to save a movdqa
    __m128i alpha2 = _mm_unpacklo_epi16(packed_alpha, packed_alpha); // 000A -> 0A0A.  Avoids a movdqa vs. shift+or, and avoids port0

    __m128i rb = _mm_and_si128(argb, _mm_set1_epi16(0x00ff));   // 0R0B
    __m128i rbmul = _mm_mullo_epi16(rb, alpha2);                // rrbb

    __m128i gmuldiv  = _mm_add_epi16(gmul, _mm_set1_epi16(1));
    __m128i rbmuldiv = _mm_add_epi16(rbmul, _mm_set1_epi16(1));  // can't overflow because rbmul can't be 0xffff
    gmuldiv  = _mm_mulhi_epu16(gmuldiv, _mm_set1_epi16(0x0101));
    rbmuldiv = _mm_mulhi_epu16(rbmuldiv, _mm_set1_epi16(0x0101)); // n/255 = ((n+1)*257) >> 16
    __m128i premul_argb = _mm_or_si128(rbmuldiv, _mm_slli_epi32(gmuldiv, 8));
    // The g vector only uses half its elements.  Could pack g together from 2 consecutive vectors.
    // especially if uinsg pack(a,a) -> unpack instead of OR(a,a<<8) to get alpha2

/*
    //too much shuffling vs. doing b and r  at the same time?

    __m128i lo = _mm_unpacklo_epi8(argb, _mm_setzero_si128());
    __m128i hi = _mm_unpackhi_epi8(argb, _mm_setzero_si128());

    // alpha =	  000W 000X      000Y 000Z
    // packed =   0W 0X 0Y 0Z   0W 0X 0Y 0Z
    // doubled =  0W0W 0X0X      0Y0Y 0Z0Z

    __m128i alpha = _mm_srli_epi32(argb, 24);   // maybe get it from lo and hi somehow?
    __m128i packed_alpha = _mm_packs_epi16(alpha, alpha);
    __m128i doubled_alpha = _mm_unpacklo_epi16(packed_alpha, packed_alpha);    // ARGB -> 0A0A.
    __m128i alphamul_lo = _mm_shuffle_epi32(doubled_alpha, _MM_SHUFFLE(1,0, 1,0));  // unpacklo would need a movdqa
    __m128i alphamul_hi = _mm_unpackhi_epi32(double_alpha, double_alpha);  // ARGB -> 0A0A0A0A.  Do this with PMULUDQ 0x01010101?  no, it take elements 0 and 2, unlike unpack
    lo = _mm_mullo_epi16(lo, alphamul_lo);
    hi = _mm_mullo_epi16(hi, alphamul_hi);  // * alpha

    lo = _mm_add_epi16(lo, _mm_set1_epi16(1));
    hi = _mm_add_epi16(hi, _mm_set1_epi16(1));
    lo = _mm_mulhi_epu16(lo, _mm_set1_epi16(0x0101));
    hi = _mm_mulhi_epu16(hi, _mm_set1_epi16(0x0101)); // n/255 = ((n+1)*257) >> 16

    __m128i premul_argb = _mm_packus_epi8(lo, hi);
*/
#endif
    return premul_argb;
  }
}

static inline
void putrow_simd(uint32_t *__restrict__ dst_start,
		 const uint32_t *__restrict__ src_start, size_t pixCount)
{
  // only called with pixCount >= 4, so we can always load & store a full vector

  // TODO: stay in the hard-alpha for a while without checking after finding one hard-alpha pixel.
  // They'll probably come in runs.
  const __m128i *lastsrc = reinterpret_cast<const __m128i*>(src_start+pixCount) - 1;
  const __m128i *srcp = reinterpret_cast<const __m128i*>(src_start);
  __m128i *__restrict__ dstp = reinterpret_cast<__m128i*>(dst_start);

  // Don't bother aligning dst pointer.  Probably not worth the overhead for modern CPUs, esp. with narrow rows
  for ( ;srcp <= lastsrc-3; srcp += 4, dstp += 4) {
    IACA_START;
    __m128i v = _mm_loadu_si128(srcp);
    _mm_storeu_si128(dstp, flipRGBAtoNative(v));

    v = _mm_loadu_si128(srcp+1);
    _mm_storeu_si128(dstp+1, flipRGBAtoNative(v));
    v = _mm_loadu_si128(srcp+2);
    _mm_storeu_si128(dstp+2, flipRGBAtoNative(v));
    v = _mm_loadu_si128(srcp+3);
    _mm_storeu_si128(dstp+3, flipRGBAtoNative(v));
  }
  IACA_END;
  for ( ; srcp <= lastsrc ; srcp++, dstp++) {
    __m128i v = _mm_loadu_si128(srcp);
    _mm_storeu_si128(dstp, flipRGBAtoNative(v));
  }
  if (srcp-1 != lastsrc) {
    // Do the last 16B, overlapping by however much is necessary
    __m128i v = _mm_loadu_si128(lastsrc);
    __m128i dstv = flipRGBAtoNative(v);
    _mm_storeu_si128((__m128i*)&dst_start[pixCount-4], dstv);
  }
}
#endif // __SSE2__


/* with return in the hard branch
        mov     edx, DWORD PTR 4[eax]   # 1 fused-domain uop, p23
        bswap   edx                     # 1 p15
        mov     ebx, edx                # 1 no port
        sub     edx, 1                  # 1 p0156
        ror     ebx, 8                  # 1 p06
        cmp     dl, -3
        jbe     .L2223                  # 1 p06 predicted not-taken
        cmp     ebx, -16777217          # 1 p0156
        cmovbe  ebx, esi                # 2  2p0156.  IACA says cmovg might be 3 uops on HSW?
        mov     DWORD PTR 4[ecx], ebx   # 1  p237 p4
   10 fused-domain uops
   1 p15
   2 p06
   4 any-port
   LEA with 2 components runs on p15, so changing mov+sub to LEA would be a win, silly compiler
*/

/****** scalar fallback, also used for dst width too narrow for SIMD  *******/
static inline void copy1PixRGBAtoNative(uint32_t *__restrict__ dst, const uint32_t *__restrict__ src)
{
  uint32_t bigendian_rgba = *src;   // low byte in memory = R, high = A
  uint32_t pix_rgba = be32toh(bigendian_rgba);   // native endian, MSB=R, LSB=A
  // uint8_t a = pix_rgba;

  uint8_t aP1 = pix_rgba + 1;  // 0 -> 1,  and 0xff -> 0.  Try to hand-hold the compiler into reusing flags for jcc and cmov

//  uint8_t a = reinterpret_cast<const uint8_t*>(src)[3];
  uint32_t pix_argb = rotr32(pix_rgba, 8);

  // Performance optimization: fully transparent/opaque pixels can be
  // processed more efficiently.
  //if (likely(a == 0 || a == 255)) {
  //if (likely(uint8_t(aM1) >= (uint8_t)0xfe)) {
  if (likely(uint8_t(aP1) <= 1)) {
  //if (likely(pix_argb <= 0x00ffffff || pix_argb >= 0xff000000)) {
    //pix_argb = (a==0) ? 0 : pix_argb;
    //uint32_t pix_argb = (a!=0) ? rotr32(pix_rgba, 8) : 0;  // extra MOV before the cmov with gcc

    //pix_argb = (a!=0) ? pix_argb : 0;  // still an extra MOV
    //pix_argb = (aM1==(uint8_t)0xfe) ? pix_argb : 0;  // reuse flags from the branch: a-1 is 255 if a==0
    pix_argb = (aP1==(uint8_t)1) ? 0 : pix_argb;  // WORKING: reuses flags from the branch. a+1 == 1 means a==0.  Good output with slow-path using swapped+rotated, forcing compiler to ror before cmp+branch, otherwise it would clobber flags with that.
    // clang still does a very bad job of this for x86 :/

    //pix_argb = (pix_argb >= 0xff000000) ? pix_argb : 0;

//    pix_argb = (pix_argb & (1UL<<31)) ? pix_argb : 0;  // A really clever x86 compiler could use CF as set by ROR to generate flags for CMOV.  But gcc TESTs the ROR output, making a longer dependency chain

    *dst = pix_argb;
  } else {
    // return;
    // FIXME: don't shadow a in the outer scope if re-enabled
    uint8_t a = reinterpret_cast<const uint8_t*__restrict__>(src)[3];  // the likely case was doing an extra MOV
    //uint8_t a = aM1 + 1;
//    uint8_t a = aP1 - 1;  // this hurts the fast-case on ARM: gcc does a uxtb to isolate a separately from adding

    // premultiply alpha
#if 1 // swapped and rotated
    uint8_t b = pix_argb;
    uint8_t g = pix_argb >> 8;
    uint8_t r = pix_argb >> 16;
#elif 0  // only swapped
    uint8_t b = pix_rgba >> 8;
    uint8_t g = pix_rgba >> 16;
    uint8_t r = pix_rgba >> 24;
#elif 1  // reload
    uint8_t r = reinterpret_cast<const uint8_t*__restrict__>(src)[0];  // reload is better on x86 where it takes 2 instructions to copy+shift a register
    uint8_t g = reinterpret_cast<const uint8_t*__restrict__>(src)[1];
    uint8_t b = reinterpret_cast<const uint8_t*__restrict__>(src)[2];
#endif

    b = uint32_t(a * b) / uint16_t(255);
    g = uint32_t(a * g) / uint16_t(255);  // uint32_t uses a longer constant, and a 64-bit imul.  But uint16_t actually uses a movzx because gcc doesn't know that uint8 * uint8 doesn't leave high garbage?
    r = uint32_t(uint16_t(a) * uint16_t(r)) / uint16_t(255);

    uint8_t *__restrict__ pix_bytes = reinterpret_cast<uint8_t*__restrict__>(dst);
    // FIXME: portability to non-GNU
    // TODO: check that portable_endian.h defines macros in all cases
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    pix_bytes[3] = a;
    #warning using LE strategy in putimage
    pix_bytes[0] = b;  // gcc calculates in the order it stores, and b first saves insns because b is easier to get at
    pix_bytes[1] = g;
    pix_bytes[2] = r;
#elif __BTYE_ORDER__ == __ORDER_BIG_ENDIAN__
    pix_bytes[0] = a;
    #warning using BE strategy in putimage
    pix_bytes[3] = b;
    pix_bytes[2] = g;
    pix_bytes[1] = r;
#else
    #warning using shift+OR native endian strategy in putimage
    (void) pix_bytes;  // unused
    uint32_t premultiplied_argb = a<<24 | r<<16 | g<<8 | b;
    *dst = premultiplied_argb;  // store in native byte order, whatever that is
#endif
  }
}

static inline
void putrow_scalar(uint32_t *__restrict__ dst, const uint32_t *__restrict__ src, size_t pixCount)
{
//  const uint32_t *lastsrc = src + pixCount - 1;
  const uint32_t *__restrict__ srcp = src;
  uint32_t *__restrict__ dstp = dst;

//#ifndef HAVE_SIMD_PUTIMAGE  // scalar version only used for rows of <4 pixels
//  for (; srcp <= lastsrc-3;  srcp+=4, dstp+=4) {
  //int signed_count = pixCount;
  if (pixCount >= 4) {  // no joy with a while() loop.
    do {
      IACA_START;
      copy1PixRGBAtoNative(dstp,   srcp);
      copy1PixRGBAtoNative(dstp+1, srcp+1);
      copy1PixRGBAtoNative(dstp+2, srcp+2);
      copy1PixRGBAtoNative(dstp+3, srcp+3);
      srcp+=4, dstp+=4;
      pixCount-=4;
    }while((ssize_t)pixCount >= 0); // works but ugly.  Also, pointer-compare only needs to macro-fuse cmp, not sub/js.  sub/js doesn't macro-fuse on SnB-family.
    // C++ makes it hard to express sub/jnc in a compiler-friendly way that won't compile to garbage if the compiler doesn't figure it out
  }
  IACA_END;
//#endif
//  for (; srcp <= lastsrc; srcp+=1, dstp+=1) {
  while((ssize_t)pixCount > 0) {
    IACA_START;
    copy1PixRGBAtoNative(dstp,   srcp);
    srcp++, dstp++;
    pixCount--;
  }
  IACA_END;
}








/********************** GetImage: shuffle and de-multiply alpha ***************************/

#ifdef __SSE2__  // x86 SIMD
#define HAVE_SIMD_GETIMAGE 1
// x86 is little endian, so we're going from BGRA to RGBA (in ascending memory addresses).
// i.e. we have to swap B with R, and keep G and A in the same place
static inline __m128i flipNativetoRGBA(__m128i argb)
{

#if defined( __SSSE3__) || defined(__AVX__)  // MSVC doesn't define __SSSE3__, only AVX if enabled
  // If this is inside the if(), gcc uses multiple copies of the shuffle constant when unrolling
  __m128i abgr = _mm_shuffle_epi8(argb, _mm_setr_epi8(2,1,0,3, 6,5,4,7, 10,9,8,11, 14,13,12,15));
#else
  __m128i word_swapped = _mm_shufflelo_epi16(argb,         _MM_SHUFFLE(2,3, 0,1));
  word_swapped         = _mm_shufflehi_epi16(word_swapped, _MM_SHUFFLE(2,3, 0,1));
  __m128i abgr = _mm_or_si128(_mm_and_si128(argb,         _mm_set1_epi16(0xff00)),   // keep G and A
			      _mm_and_si128(word_swapped, _mm_set1_epi16(0x00ff)) ); // keep R and B
  // XOR, AND, XOR  could blend with only one mask constant, but less ILP.
  // andnot to share the mask would require an extra movdqa, because it destroys the inverted operand.
#endif
  // alpha == 255 || alpha == 0 can be done with  unsigned(alpha - 1) > 253
  // but SSE2 only has signed comparisons, so we do  unsigned(alpha - 1) - 128 > (253-128)
  __m128i alpha_check = _mm_sub_epi8(argb, _mm_set1_epi8(uint8_t(129)));
  alpha_check = _mm_cmpgt_epi8(alpha_check, _mm_set1_epi8(253-128));
  unsigned easy_alpha = _mm_movemask_ps(_mm_castsi128_ps(alpha_check));  // grab high bits of just the alpha pixels

  if (likely(easy_alpha == 0b1111)) { // all 4 elements have easy alpha
    return abgr;
  } else {
    //__m128i abgr = argb;      // FIXME: experiment only
    //__m128i argb = abgr;      // FIXME: experiment
    // SSE2 can use argb and abgr because pshufl/hw is a copy+shuffle
    // but for SSSE3 that slows down the fast case because pshufb is destructive

    // if (easy_alpha != 0)  // store all and then scalar store just the hard ones?
    // unpacking 4 vectors to get vectors of all-red, all-green, and all-blue might make sense.
    // (one non-easy alpha probably means lots of non-easy alpha)

    // TODO: optimize this; too much masking and shifting.  Maybe some shuffles would help?
    // Maybe convert to float without shifting to the bottom?  float32 can exactly represent multiples of 2^24.
    // That introduces double-rounding when rounding back to integer*256 and then truncating with a mask.

    // Also, the high bit is the sign bit, so we can't do that for alpha
//    __m128i alpha_isolated = _mm_and_si128(abgr, _mm_set1_epi32(0xFF000000));  // bug: treats alpha as signed
//    __m128 inv_alpha = _mm_div_ps(_mm_set1_ps(255.0f * (float)(1<<24)), _mm_cvtepi32_ps(alpha_isolated));

    // TODO: try unpack/pack for less shifting (port0 pressure)?
    __m128i alpha_int = _mm_srli_epi32(argb, 24);
    __m128 inv_alpha = _mm_div_ps(_mm_set1_ps(255.0f), _mm_cvtepi32_ps(alpha_int));
    __m128 rf = _mm_cvtepi32_ps(_mm_and_si128(argb, _mm_set1_epi32(0x000000FF)));
    __m128 gf = _mm_cvtepi32_ps(_mm_and_si128(_mm_srli_epi32(argb, 8), _mm_set1_epi32(0x000000FF)));
    __m128 bf = _mm_cvtepi32_ps(_mm_and_si128(_mm_srli_epi32(argb, 16), _mm_set1_epi32(0x000000FF)));
    rf = _mm_mul_ps(rf, inv_alpha);
    gf = _mm_mul_ps(gf, inv_alpha);
    bf = _mm_mul_ps(bf, inv_alpha);
    __m128i r = _mm_cvttps_epi32(rf);  // nearest instead of truncate would preserves values for a round-trip.
    __m128i g = _mm_cvttps_epi32(gf);  // But old behaviour is truncate.
    __m128i b = _mm_cvttps_epi32(bf);
    __m128i rg  = _mm_or_si128(r,  _mm_slli_epi32(g, 8));
    __m128i rgb = _mm_or_si128(rg, _mm_slli_epi32(b, 16));
    __m128i demultiplied_abgr = _mm_or_si128(rgb, _mm_slli_epi32(alpha_int, 24));  // or mask abgr
    return demultiplied_abgr;
  }
}

static inline void getrow_simd(uint32_t *__restrict__ dst_start,
					       const uint32_t *__restrict__ src_start, size_t pixCount)
{
  // only called with pixCount >= 4, so we can always load & store a full vector

  // TODO: gather src pixels into vectors (PALIGNR) so we can still flip them with SIMD for small dst widths
  // Or just read outside the copy-source region, as long as it's not at the bottom-right corner (outside the canvas)
  // overlapping stores to dst are fine, too, so we can overshoot as long as it's not the last row

  const __m128i *lastsrc = reinterpret_cast<const __m128i*>(src_start+pixCount) - 1;
  const __m128i *srcp = reinterpret_cast<const __m128i*>(src_start);
  __m128i *__restrict__ dstp = reinterpret_cast<__m128i*>(dst_start);

  // Don't bother aligning dst pointer.  Probably not worth the overhead for modern CPUs, esp. with narrow rows
  while (srcp <= lastsrc-3) {
    IACA_START;
    __m128i v = _mm_loadu_si128(srcp);
    _mm_storeu_si128(dstp, flipNativetoRGBA(v));

    v = _mm_loadu_si128(srcp+1);
    _mm_storeu_si128(dstp+1, flipNativetoRGBA(v));
    v = _mm_loadu_si128(srcp+2);
    _mm_storeu_si128(dstp+2, flipNativetoRGBA(v));
    v = _mm_loadu_si128(srcp+3);
    _mm_storeu_si128(dstp+3, flipNativetoRGBA(v));
    srcp += 4;
    dstp += 4;
  }
  IACA_END;
  for ( ; srcp <= lastsrc ; srcp++, dstp++) {
    __m128i v = _mm_loadu_si128(srcp);
    _mm_storeu_si128(dstp, flipNativetoRGBA(v));
  }
  if (srcp-1 != lastsrc) {
    // Do the last 16B, overlapping by however much is necessary
    __m128i v = _mm_loadu_si128(lastsrc);
    __m128i dstv = flipNativetoRGBA(v);
    _mm_storeu_si128((__m128i*)&dst_start[pixCount-4], dstv);
  }
}
#endif // __SSE2__


/***** scalar fallback, also used for dst width too narrow for SIMD  ******/
static inline void copy1PixNativetoRGBA(uint32_t *dst, const uint32_t *src)
{
  uint32_t pix_argb = *src;  // native endian, MSB=A, LSB=B
  uint32_t pix_rgba = rotl32(pix_argb, 8); // native, MSB=R, LSB=A
  uint8_t a = pix_rgba;        // LSB=A

  // Performance optimization: fully transparent/opaque pixels can be
  // processed more efficiently.
  if (likely(a == 0 || a == 255)) {
    uint32_t bigendian_rgba = htobe32(pix_rgba);  // first byte in memory = R, last = A
    *dst = bigendian_rgba;
  } else {
    uint8_t r = pix_rgba >> 24;  // use the already-rotated data to help the compiler optimize the easy-alpha case better
    uint8_t g = pix_rgba >> 16;
    uint8_t b = pix_rgba >> 8;
    uint8_t *pix_bytes = reinterpret_cast<uint8_t*>(dst);
    pix_bytes[3] = a;

#if 0 // integer division is much worse throughput than FP multiplication
    pix_bytes[0] = uint32_t(uint16_t(255) * uint16_t(r)) / uint16_t(a);
    pix_bytes[1] = uint32_t(255 * g) / uint16_t(a);
    pix_bytes[2] = uint16_t(255 * b) / uint16_t(a);  // 16-bit idiv isn't much if any faster on modern x86, but IDK about ARM
#else
    // TODO: test float vs. integer on 32-bit no-SSE2 x86 maybe?  Or ARM?  x86-64 will always use SSE2
    // 32-bit x86 Without SSE has to use x87, so int->float->int is a lot of store/reloads
    // gcc -m32 defaults to SSE2 enabled (but FP math using x87)
    // so most x86-32 builds will use this only for very narrow dst
    float inv_alpha = 255.0f / (float)a;
    pix_bytes[0] = static_cast<int>(r * inv_alpha); // TODO: round to nearest instead of truncating?
    pix_bytes[1] = static_cast<int>(g * inv_alpha);
    pix_bytes[2] = static_cast<int>(b * inv_alpha);
#endif
  }
}

// TODO: share loop code with putrow_scalar, and use cleaner array indexing
static inline void getrow_scalar(uint32_t *dst, const uint32_t *src, size_t pixCount)
{
  size_t x;
  for (x = 0; x < pixCount-3; x+=4) {
    copy1PixNativetoRGBA(&dst[x],   &src[x]);
    copy1PixNativetoRGBA(&dst[x+1], &src[x+1]);
    copy1PixNativetoRGBA(&dst[x+2], &src[x+2]);
    copy1PixNativetoRGBA(&dst[x+3], &src[x+3]);
  }
  for (; x < pixCount; x++) {
    copy1PixNativetoRGBA(&dst[x],   &src[x]);
  }
}







/*
  Putimage narrow-row strategies: srcstride may == cols, but putimage supports copying only part of ImageData

    3: palignr loads?  Maybe to flipped vectors so we can store with movhps / movd?  else movq / punpckhqdq / movd
    2: movq / movhps to load and store
    1: load with 2x(movd/shufps) -> por, instead of punpcklqdq(2x(movd/punpckldq))
    Store with movd / psrldq?   Easier at right edge if we can write into padding with movhps

// getimage: dststride == cols, so we can gather full vectors and store them
    3: palignr loads, movdqu stores
    2: movq / movhps loads, movdqu stores
    1: gather in-order is important, 4 movd + 3 shuffles.
*/


// TODO: factor out the duplication of looping with a put/get template function?
// big-endian RGBA -> alpha-premultiplied native-endian ARGB
void PutPixels(uint32_t *__restrict__ dst, const uint32_t *__restrict__ src,
	       size_t rows, size_t cols, size_t dstPixStride, size_t srcPixStride)
{

//  if (rows < 1 || cols < 1)  // just use do{}while() loops
//    __builtin_unreachable();

  // src stride == width is not guaranteed; PutImageData allows sub-rectangles
  if (srcPixStride == cols &&
      srcPixStride == dstPixStride) {
    // treat as one giant row instead of actually inlining another copy of the unrolled loop.
    cols *= rows;
    rows = 1;
    //dstPixStride = srcPixStride = cols;  // unused
  }

  // TODO: allow writing past width out to stride, to scribble on cairo's row padding
  //  when that's convenient for copying a non-multiple-of-4 width to the right edge of the surface
#ifdef HAVE_SIMD_PUTIMAGE
  if (cols >= 4) {
    size_t y = 0;
    do {  // caller guarantees that rows > 0 && cols > 0
      putrow_simd(dst, src, cols);
      dst += dstPixStride;
      src += srcPixStride;
    } while(++y < rows);
    return;
  }
#endif

  size_t y = 0;
  do {
    putrow_scalar(dst, src, cols);
    dst += dstPixStride;
    src += srcPixStride;
  } while (++y < rows);
}

// alpha-premultiplied native-endian ARGB -> big-endian RGBA (de-multiply alpha)
void GetPixels(uint32_t *__restrict__ dst, const uint32_t *__restrict__ src,
	       size_t rows, size_t cols, size_t dstPixStride, size_t srcPixStride)
{
  // src stride == width is guaranteed for GetImageData, but check anyway I guess.  It can optimize away with LTO
  if (srcPixStride == cols &&
      srcPixStride == dstPixStride) {
    // treat as one giant row instead of actually inlining another copy of the unrolled loop.
    cols *= rows;
    rows = 1;
    //dstPixStride = srcPixStride = cols;
  }

#ifdef HAVE_SIMD_GETIMAGE
  if (cols >= 4) {
    for (size_t y = 0; y < rows; ++y) {   // TODO: do{}while
      getrow_simd(dst, src, cols);
      dst += dstPixStride;
      src += srcPixStride;
    }
    return;
  }
#endif
  // TODO: do{}while
  for (size_t y = 0; y < rows; ++y) {
    getrow_scalar(dst, src, cols);
    dst += dstPixStride;
    src += srcPixStride;
  }
}
