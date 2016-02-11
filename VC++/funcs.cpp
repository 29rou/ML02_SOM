#include "som.h"

void initialize(imgdata *src, somap *dst) {
	dst->img = src->img;
	*dst->fvex = *src->fvex;
	//cout << dst->fvex<<" "<<src->fvex<<endl;
}

float operator-(const imgdata &obj1, const somap &obj2) {
	float r = 0;
	__m256 tmp[f];
	//__m256 tmp2[f];
	__m256 *v1 = (__m256*)(obj1.fvex);
	__m256 *v2 = (__m256*)(obj2.fvex);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
	for (int i = 0; i <f; i++) {
		//cout << i << endl;
		tmp[i] = _mm256_sub_ps(*v1, *v2);
		//tmp2[i] = _mm256_mul_ps(tmp[i], tmp[i]);
		int rtmp = fabs(tmp[i].m256_f32[0]);
		rtmp += fabs(tmp[i].m256_f32[1]);
		rtmp += fabs(tmp[i].m256_f32[2]);
		rtmp += fabs(tmp[i].m256_f32[3]);
		rtmp += fabs(tmp[i].m256_f32[4]);
		rtmp += fabs(tmp[i].m256_f32[5]);
		rtmp += fabs(tmp[i].m256_f32[6]);
		rtmp += fabs(tmp[i].m256_f32[7]);
#pragma omp atomic
		r += rtmp;
#pragma omp atomic
		v1++;
#pragma omp atomic
		v2++;
	}
/*#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int i = 0; i < f; i++) {
		int rtmp = tmp2[i].m256_f32[0];
		rtmp += tmp2[i].m256_f32[1];
		rtmp += tmp2[i].m256_f32[2];
		rtmp += tmp2[i].m256_f32[3];
		rtmp += tmp2[i].m256_f32[4];
		rtmp += tmp2[i].m256_f32[5];
		rtmp += tmp2[i].m256_f32[6];
		rtmp += tmp2[i].m256_f32[7];
		
#pragma omp atomic
		r += rtmp;
	}*/
	//_m256 dst = _mm256_sub_ps((obj1.fvex));
	return r;//sqrtf(r);
}

somap* findnear(const imgdata *imgd, somap *smp) {
	random_device rnd;
	mt19937_64 mt(rnd());
	float min = FLT_MAX;
	float tmp;
	vector<int> is;
	for (int i = 0; i < HW; i++) {
		tmp = *imgd - smp[i];
		if (tmp <= min) {
			min = tmp;
			is.push_back(i);
		}
	}
	if (is.empty())return 0;
	uniform_int_distribution<> randin(0, (is.size() - 1));
	somap* num = smp + is[randin(mt)];
	return num;
}