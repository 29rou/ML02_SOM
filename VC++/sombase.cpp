
#include "som.h"

std::random_device sombase::rnd;
std::mt19937_64 sombase::mt(rnd());

sombase::sombase()
{
	this->fvex = (float*)_aligned_malloc(sizeof(float)*F256, 32);
	__m256 *v = (__m256*)(this->fvex);
	for (int i = 0; i < f;i++)v[i]= _mm256_set1_ps(0);
}


sombase::~sombase()
{
	_aligned_free(this->fvex);
}

const float sombase::getDistance(const sombase & obj)
{
	const __m256 *v1 = (__m256*)(this->fvex);
	const __m256 *v2 = (__m256*)(obj.fvex);
	__m256 tmp;
	__m256 sum = _mm256_set1_ps(0.0);
	static const __m256 signmask = _mm256_set1_ps(-0.0f);
	for (int i = 0; i < f; i++) {
		tmp = _mm256_sub_ps(v1[i], v2[i]);
		tmp = _mm256_andnot_ps(signmask, tmp);
		sum = _mm256_add_ps(sum, tmp);
	}
	for (int i = 1; i < 8; i++)sum.m256_f32[0] += sum.m256_f32[i];
	return sum.m256_f32[0];
}
