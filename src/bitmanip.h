#ifndef BITMANIP_H__
#define BITMANIP_H__

#ifndef TEST_FREESTANDING   // FIXME: only for asm output with arm-none
#include <limits.h> // for CHAR_BIT
#else
#define CHAR_BIT 8 // FIXME: don't commit this
typedef unsigned long size_t; // FIXME
#endif

// http://stackoverflow.com/questions/776508/best-practices-for-circular-shift-rotate-operations-in-c
static inline uint32_t rotl32 (uint32_t n, unsigned int c)
{
  const unsigned int mask = (CHAR_BIT*sizeof(n)-1);
  c &= mask;
  return (n<<c) | (n>>( (-c)&mask ));
}

static inline uint32_t rotr32 (uint32_t n, unsigned int c)
{
  const unsigned int mask = (CHAR_BIT*sizeof(n)-1);
  c &= mask;
  return (n>>c) | (n<<( (-c)&mask ));
}




// from public-domain portable_endian.h  https://gist.github.com/PkmX/63dd23f28ba885be53a5
#if (defined(_WIN16) || defined(_WIN32) || defined(_WIN64)) && !defined(__WINDOWS__)
#	define __WINDOWS__
#endif

#if defined(__linux__) || defined(__CYGWIN__)
#	include <endian.h>

#elif defined(__GNUC__)  // gcc, clang, icc, etc. all support the GNU dialects of C / C++
// The GNU builtins always inline, and the compiler understands what they do.
// Prefer these over platform-detection stuff.

#	ifndef BYTE_ORDER
#		define BYTE_ORDER __BYTE_ORDER__
#		define LITTLE_ENDIAN __ORDER_LITTLE_ENDIAN__
#		define BIG_ENDIAN __ORDER_BIG_ENDIAN__
#	endif

#	if BYTE_ORDER == LITTLE_ENDIAN     // any GNU C compiler, any platform, little endian
#		define htobe16(x) __builtin_bswap16(x)
#		define htole16(x) (x)
#		define be16toh(x) __builtin_bswap16(x)
#		define le16toh(x) (x)

#		define htobe32(x) __builtin_bswap32(x)
#		define htole32(x) (x)
#		define be32toh(x) __builtin_bswap32(x)
#		define le32toh(x) (x)

#		define htobe64(x) __builtin_bswap64(x)
#		define htole64(x) (x)
#		define be64toh(x) __builtin_bswap64(x)
#		define le64toh(x) (x)
#	elif BYTE_ORDER == BIG_ENDIAN    // any GNU C compiler, any platform, big endian
#		define htobe16(x) (x)
#		define htole16(x) __builtin_bswap16(x)
#		define be16toh(x) (x)
#		define le16toh(x) __builtin_bswap16(x)

#		define htobe32(x) (x)
#		define htole32(x) __builtin_bswap32(x)
#		define be32toh(x) (x)
#		define le32toh(x) __builtin_bswap32(x)

#		define htobe64(x) (x)
#		define htole64(x) __builtin_bswap64(x)
#		define be64toh(x) (x)
#		define le64toh(x) __builtin_bswap64(x)
#	else
#		error byte order not supported
#	endif

#elif defined(__APPLE__)
#	include <libkern/OSByteOrder.h>
#	define htobe16(x) OSSwapHostToBigInt16(x)
#	define htole16(x) OSSwapHostToLittleInt16(x)
#	define be16toh(x) OSSwapBigToHostInt16(x)
#	define le16toh(x) OSSwapLittleToHostInt16(x)

#	define htobe32(x) OSSwapHostToBigInt32(x)
#	define htole32(x) OSSwapHostToLittleInt32(x)
#	define be32toh(x) OSSwapBigToHostInt32(x)
#	define le32toh(x) OSSwapLittleToHostInt32(x)

#	define htobe64(x) OSSwapHostToBigInt64(x)
#	define htole64(x) OSSwapHostToLittleInt64(x)
#	define be64toh(x) OSSwapBigToHostInt64(x)
#	define le64toh(x) OSSwapLittleToHostInt64(x)

#	define __BYTE_ORDER    BYTE_ORDER
#	define __BIG_ENDIAN    BIG_ENDIAN
#	define __LITTLE_ENDIAN LITTLE_ENDIAN
#	define __PDP_ENDIAN    PDP_ENDIAN

#elif defined(__OpenBSD__)
#	include <sys/endian.h>

#elif defined(__NetBSD__) || defined(__FreeBSD__) || defined(__DragonFly__)
#	include <sys/endian.h>
#	define be16toh(x) betoh16(x)
#	define le16toh(x) letoh16(x)

#	define be32toh(x) betoh32(x)
#	define le32toh(x) letoh32(x)

#	define be64toh(x) betoh64(x)
#	define le64toh(x) letoh64(x)

#elif defined(__WINDOWS__)
#	include <windows.h>
// MSVC doesn't inline ntohl, so we need to use compiler intrinsics to avoid a DLL call instead of one instruction
#	if BYTE_ORDER == LITTLE_ENDIAN

#               if defined(_MSC_VER)    // Windows MSVC little endian
#                       include <stdlib.h>
#			define htobe16(x) _byteswap_ushort(x)
#			define htole16(x) (x)
#			define be16toh(x) _byteswap_ushort(x)
#			define le16toh(x) (x)

#			define htobe32(x) _byteswap_ulong(x)
#			define htole32(x) (x)
#			define be32toh(x) _byteswap_ulong(x)
#			define le32toh(x) (x)

#			define htobe64(x) _byteswap_uint64(x)
#			define htole64(x) (x)
#			define be64toh(x) _byteswap_uint64(x)
#			define le64toh(x) (x)

#               elif defined(__GNUC__) || defined(__clang__)   // Windows GNU C, little endian
#			define htobe16(x) __builtin_bswap16(x)
#			define htole16(x) (x)
#			define be16toh(x) __builtin_bswap16(x)
#			define le16toh(x) (x)

#			define htobe32(x) __builtin_bswap32(x)
#			define htole32(x) (x)
#			define be32toh(x) __builtin_bswap32(x)
#			define le32toh(x) (x)

#			define htobe64(x) __builtin_bswap64(x)
#			define htole64(x) (x)
#			define be64toh(x) __builtin_bswap64(x)
#			define le64toh(x) (x)
#               else
#                       error platform not supported
#               endif

#	elif BYTE_ORDER == BIG_ENDIAN
		/* that would be xbox 360 */
#               if defined(_MSC_VER)       // Windows MSVC big endian
#                       include <stdlib.h>
#			define htobe16(x) (x)
#			define htole16(x) _byteswap_ushort(x)
#			define be16toh(x) (x)
#			define le16toh(x) _byteswap_ushort(x)

#			define htobe32(x) (x)
#			define htole32(x) _byteswap_ulong(x)
#			define be32toh(x) (x)
#			define le32toh(x) _byteswap_ulong(x)

#			define htobe64(x) (x)
#			define htole64(x) _byteswap_uint64(x)
#			define be64toh(x) (x)
#			define le64toh(x) _byteswap_uint64(x)
#		elif defined(__GNUC__) || defined(__clang__) // Windows GNU C big endian
#			define htobe16(x) (x)
#			define htole16(x) __builtin_bswap16(x)
#			define be16toh(x) (x)
#			define le16toh(x) __builtin_bswap16(x)

#			define htobe32(x) (x)
#			define htole32(x) __builtin_bswap32(x)
#			define be32toh(x) (x)
#			define le32toh(x) __builtin_bswap32(x)

#			define htobe64(x) (x)
#			define htole64(x) __builtin_bswap64(x)
#			define be64toh(x) (x)
#			define le64toh(x) __builtin_bswap64(x)
#               else
#                       error platform not supported
#               endif
#	else  // Windows unknown endian
#		error byte order not supported
#	endif

#	define __BYTE_ORDER    BYTE_ORDER
#	define __BIG_ENDIAN    BIG_ENDIAN
#	define __LITTLE_ENDIAN LITTLE_ENDIAN
#	define __PDP_ENDIAN    PDP_ENDIAN

#else
#	ifdef __unix__
#	include <unistd.h>  // for _POSIX_VERSION
#	else
#		error platform not supported
#	endif

#	ifdef _POSIX_VERSION
#		include <arpa/inet.h>
#       	if defined(BYTE_ORDER) && defined(LITTLE_ENDIAN)
#			if BYTE_ORDER == LITTLE_ENDIAN
#				define htobe16(x) htons(x)
#				define htole16(x) (x)
#				define be16toh(x) ntohs(x)
#				define le16toh(x) (x)

#				define htobe32(x) htonl(x)
#				define htole32(x) (x)
#				define be32toh(x) ntohl(x)
#				define le32toh(x) (x)

#				define htobe64(x) htonll(x)
#				define htole64(x) (x)
#				define be64toh(x) ntohll(x)
#				define le64toh(x) (x)
#			else
				// to/from le isn't possible with no-op ntoh functions on a big-endian platform
#				error "only little-endian hosts supported with generic ntohl"
#			endif
#		else
#			error "platform doesn't define BYTE_ORDER detection macros"
#		endif
#	else // __unix__ but non-POSIX
#		error "don't know how to byte-swap on this non-POSIX unix"
#	endif
#endif  // OSes



#endif  // BITMANIP_H__
