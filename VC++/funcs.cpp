#include "som.h"

somap* initialize(imgdata *imgd) {
	somap *ptr = new somap[HW];
	vector<int> chk;
	int *tmp = new int[HW];
#ifdef _OPENMP
	omp_set_num_threads(8);
#pragma omp parallel for schedule(static)
#endif
	for (int i = 0; i < HW; i++) {
		do {
			tmp[i] = imgd->rand();
		} while (find(chk.begin(), chk.end(), tmp[i]) != chk.end());
		for (int k = 0; k < F; k++) {
			if (isinf(imgd[tmp[i]].fvex[k])) {
				ptr[i].fvex[k] = FLT_MAX;
			}
			else if (isnan(imgd[tmp[i]].fvex[k])) {
				ptr[i].fvex[k] = 0;
			}
			else ptr[i].fvex[k] = imgd[tmp[i]].fvex[k];
		}
#pragma omp critical
		chk.push_back(tmp[i]);
	}
	delete[] tmp;
	return ptr;
}

Mat* toimg(imgdata* imgd,somap *smp, Mat* combined_img) {
	Rect roi_rect;
	roi_rect.width = WIDTH;
	roi_rect.height = HEIGHT;
	for (int i = 0; i <HW; i++) {
		if ((smp[i].x % 3 == 1) && (smp[i].y % 3 == 1)) {
			Mat roi(*combined_img, roi_rect);
			smp[i].picimg(imgd)->copyTo(roi);
			roi_rect.x += WIDTH;
		}
		if ((smp[i].x == W-1) && (smp[i].y % 3 == 1)) {
			roi_rect.x = 0;
			roi_rect.y += HEIGHT;
		}
	}
	return combined_img;
}

float operator-(const imgdata &obj1, const somap &obj2) {
	__m256 *tmp = (__m256*)_aligned_malloc(sizeof(__m256)*f, 32);
	__m256 *v1 = (__m256*)(obj1.fvex);
	__m256 *v2 = (__m256*)(obj2.fvex);
	const __m256 signmask = _mm256_set1_ps(-0.0f); // 0x80000000
#ifdef _OPENMP
#pragma omp parallel
#endif
	{
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
		for (int i = 0; i < f; i++)tmp[i] = _mm256_sub_ps(*(v1 + i), *(v2 + i));
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
		for (int i = 0; i < f; i++)tmp[i] = _mm256_andnot_ps(signmask, tmp[i]);
	}
	__m256 sum = _mm256_set1_ps(0);
	for (int i = 0; i < f; i++) {
		sum = _mm256_sub_ps(sum, tmp[i]);
	}
	float r=0;
#pragma omp parallel for reduction(+:r) 
	for (int i = 0; i < 8; i++)r += sum.m256_f32[i];
	_aligned_free(tmp);
	return r;
}