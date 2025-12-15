#pragma once

/*
template <typename T = uint8_t>
float compute_distance_squared(int dim, const T* __restrict__ a, const T* __restrict__ b) {
  if constexpr(std::is_same<T, uint8_t>::value) {// for short integer like uint8 or uint16
    int ans = 0;
    #pragma omp simd
    for(int i = 0;i < dim; ++ i)
        ans += (int(a[i]) - int(b[i])) * (int(a[i]) - int(b[i]));
    return float(ans);
  } else {
    T ans = 0;
    #pragma omp simd
    for(int i = 0;i < dim; ++ i)
        ans += (a[i] - b[i]) * (a[i] - b[i]);
    return float(ans);
  }
}
*/
int compute_distance_squared(int dim, const uint8_t* __restrict__ a, const uint8_t* __restrict__ b) {
    int ans = 0;
    #pragma omp simd reduction(+ : ans) aligned(a, b : 8)
    for(int i = 0;i < dim; ++ i)
        ans += (int(a[i]) - int(b[i])) * (int(a[i]) - int(b[i]));
    return ans;
}

/*
template <typename T = uint8_t>
int compute_distance_squared(int dim, const T* __restrict__ a, const T* __restrict__ b) {
  int ans = 0;
  #pragma omp simd
  for(int i = 0;i < dim; ++ i)
    ans += (int(a[i]) - int(b[i])) * (int(a[i]) - int(b[i]));
  return ans;
}
*/

/*
template <typename T = float>
T compute_distance_squared(int dim, const T* __restrict__ a, const T* __restrict__ b) {
  T ans = 0.;
  #pragma omp simd
  for(int i = 0;i < dim; ++ i)
    ans += (a[i] - b[i]) * (a[i] - b[i]);
  return ans;
}
*/
/*
template <typename T = uint8_t>
int compute_ip_distance(int dim, const T* __restrict__ a, const T* __restrict__ b) {
  if constexpr(std::is_same<T, uint8_t>::value) {// for short integer like uint8 or uint16
    int ans = 0;
    #pragma omp simd
    for(int i = 0;i < dim; ++ i)
        ans += int(a[i]) * int(b[i]);
    return -ans;
  } else {
    T ans = 0.;
    #pragma omp simd
    for(int i = 0;i < dim; ++ i)
        ans += a[i] * b[i];
    return -ans;
  }
}
*/

int compute_ip_distance(int dim, const uint8_t* __restrict__ a, const uint8_t* __restrict__ b) {
    int ans = 0;
    #pragma omp simd
    for(int i = 0;i < dim; ++ i)
        ans += int(a[i]) * int(b[i]);
    return -ans;
}

float compute_ip_distance(int dim, const float* __restrict__ a, const float* __restrict__ b) {
    float ans = 0.;
    #pragma omp simd
    for(int i = 0;i < dim; ++ i)
        ans += a[i] * b[i];
    return -ans;
}

// from DiskANN
#include <immintrin.h>
#include <cstdint>
static inline float _mm256_reduce_add_ps(__m256 x) {
    /* ( x3+x7, x2+x6, x1+x5, x0+x4 ) */
    const __m128 x128 = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    /* ( -, -, x1+x3+x5+x7, x0+x2+x4+x6 ) */
    const __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
    /* ( -, -, -, x0+x1+x2+x3+x4+x5+x6+x7 ) */
    const __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    /* Conversion to float is a no-op on x86-64 */
    return _mm_cvtss_f32(x32);
}

float compute_distance_squared(int dim, const float* __restrict__ a, const float* __restrict__ b) {
  if (((dim*sizeof(float)) % 32) == 0) {
    a = (const float *)__builtin_assume_aligned(a, 32);
    b = (const float *)__builtin_assume_aligned(b, 32);
  
    // assume size is divisible by 8
    uint16_t niters = (uint16_t)(dim / 8);
    __m256 sum = _mm256_setzero_ps();
    for (uint16_t j = 0; j < niters; j++)
    {
        // scope is a[8j:8j+7], b[8j:8j+7]
        if (j+1 < niters)
        {
            _mm_prefetch((char *)(a + 8 * (j + 1)), _MM_HINT_T0);
            _mm_prefetch((char *)(b + 8 * (j + 1)), _MM_HINT_T0);
        }
        __m256 a_vec = _mm256_load_ps(a + 8 * j);
        // load b_vec
        __m256 b_vec = _mm256_load_ps(b + 8 * j);
        // a_vec - b_vec
        __m256 tmp_vec = _mm256_sub_ps(a_vec, b_vec);

        sum = _mm256_fmadd_ps(tmp_vec, tmp_vec, sum);
    }

    // horizontal add sum
    return _mm256_reduce_add_ps(sum);
  }
  else {
    float ans = 0.;
    #pragma omp simd
    for(int i = 0;i < dim; ++ i)
        ans += (a[i] - b[i]) * (a[i] - b[i]);
    return ans;
  }
}

#include <cmath>
float compute_distance(int dim, const float* __restrict__ a, const float* __restrict__ b) {
    return sqrt(compute_distance_squared(dim,a,b));
}

/*template <typename T = float>
T compute_distance(int dim, const T* __restrict__ a, const T* __restrict__ b) {
  T ans = 0.;
  #pragma omp simd
  for(int i = 0;i < dim; ++ i)
    ans += (a[i] - b[i]) * (a[i] - b[i]);
  return std::sqrt(ans);
}*/

template <typename T = uint8_t>
int compute_distance_squared_early_stop(int dim, const T* __restrict__ a, const T* __restrict__ b, int const cutoff) {
  int ans = 0;
  //#pragma omp simd
  for(int i = 0;i < dim; ++ i)
    if((ans += (int(a[i]) - int(b[i])) * (int(a[i]) - int(b[i]))) > cutoff)
      break;
  return ans;
}

/*template <typename T = float>
T compute_distance_squared_early_stop(int dim, const T* __restrict__ a, const T* __restrict__ b, T const cutoff) {
  T ans = 0.;
  for(int i = 0;i < dim; ++ i)
    if((ans += (a[i] - b[i]) * (a[i] - b[i])) > cutoff)
      return cutoff;
  return ans;
}*/

// SIMD version
template <typename T = float>
T compute_distance_avx2(int dim, const T* __restrict__ a, const T* __restrict__ b) {
  T result = 0;
#define AVX_L2SQR(addr1, addr2, dest, tmp1, tmp2) \
      tmp1 = _mm256_loadu_ps(addr1);\
      tmp2 = _mm256_loadu_ps(addr2);\
      tmp1 = _mm256_sub_ps(tmp1, tmp2); \
      tmp1 = _mm256_mul_ps(tmp1, tmp1); \
      dest = _mm256_add_ps(dest, tmp1);

  const int N = 8;
  const int N2 = 16;
  const int NS = N-1;

  __m256 sum;
  __m256 l0, l1;
  __m256 r0, r1;
  unsigned D = (dim + NS) & ~7U;
  unsigned DR = D % N2;
  unsigned DD = D - DR;
  const T* l = a;
  const T* r = b;
  const T* e_l = l + DD;
  const T* e_r = r + DD;
  T unpack[N] __attribute__ ((aligned (32))) = {0, 0, 0, 0, 0, 0, 0, 0};

  sum = _mm256_load_ps(unpack);
  if(DR) { AVX_L2SQR(e_l, e_r, sum, l0, r0); }

  for (unsigned i = 0; i < DD; i += N2, l += N2, r += N2) {
    AVX_L2SQR(l, r, sum, l0, r0);
    AVX_L2SQR(l + N, r + N, sum, l1, r1);
  }
  _mm256_store_ps(unpack, sum);
  result = unpack[0] + unpack[1] + unpack[2] + unpack[3] + unpack[4] + unpack[5] + unpack[6] + unpack[7];

  return result;
}

template <typename T = float>
T compute_distance_avx512(int dim, const T* __restrict__ a, const T* __restrict__ b) {
  T result = 0;
#define AVX512_L2SQR(addr1, addr2, dest, tmp1, tmp2) \
      tmp1 = _mm512_loadu_ps(addr1);\
      tmp2 = _mm512_loadu_ps(addr2);\
      tmp1 = _mm512_sub_ps(tmp1, tmp2); \
      tmp1 = _mm512_mul_ps(tmp1, tmp1); \
      dest = _mm512_add_ps(dest, tmp1);

  const int N = 16;
  const int N2 = 32;
  const int NS = N-1;

  __m512 sum;
  __m512 l0, l1;
  __m512 r0, r1;
  unsigned D = (dim + NS) & ~15U;
  unsigned DR = D % N2;
  unsigned DD = D - DR;
  const T* l = a;
  const T* r = b;
  const T* e_l = l + DD;
  const T* e_r = r + DD;
  T unpack[N2] __attribute__ ((aligned (32))) = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  sum = _mm512_load_ps(unpack);
  if(DR) { AVX512_L2SQR(e_l, e_r, sum, l0, r0); }

  for (unsigned i = 0; i < DD; i += N2, l += N2, r += N2) {
    AVX512_L2SQR(l, r, sum, l0, r0);
    AVX512_L2SQR(l + N, r + N, sum, l1, r1);
  }
  _mm512_store_ps(unpack, sum);
  result = unpack[0] + unpack[1] + unpack[2] + unpack[3] + unpack[4] + unpack[5] + unpack[6] + unpack[7];
  result+= unpack[8] + unpack[9] + unpack[10] + unpack[11] + unpack[12] + unpack[13] + unpack[14] + unpack[15];

  return result;
}
