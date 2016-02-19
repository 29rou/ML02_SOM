#include "som.h"

somap* initialize(imgdata *imgd) {
	somap *ptr = new somap[HW];
	vector<int> chk;
	int tmp[HW];
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
	for (int i = 0; i < HW; i++) {
		do {
				tmp[i] = imgd->rand();
		} while (find(chk.begin(), chk.end(), tmp[i]) != chk.end());
#pragma omp critical
			chk.push_back(tmp[i]);
		__m256 *dst = (__m256*) (ptr[i].fvex);
		__m256 *src = (__m256*) (imgd[tmp[i]].fvex);
		for (int k = 0; k < f; k++) {
			dst[k] = src[k];
		}
	}
	return ptr;
}

Mat* toimg(imgdata* imgd, somap *smp, Mat* combined_img) {
	Rect roi_rect;
	roi_rect.width = W*WIDTH;
	roi_rect.height = HEIGHT;
	Mat *img[HW];
	Mat *tmp[H];
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
	for (int i=0; i < HW; i++)img[i] = smp[i].picimg(imgd);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
	for (int i = 0; i <H; i++) {
		tmp[i] = new Mat(Size(W*WIDTH, HEIGHT), CV_8UC3);
		Rect roi_tmp;
		roi_tmp.width = WIDTH;
		roi_tmp.height = HEIGHT;
		for (int j = 0; j < W; j++) {
			int index = j + i*W;
			Mat roi(*tmp[i], roi_tmp);
			img[index]->copyTo(roi);
			roi_tmp.x += WIDTH;
		}
	}
	for (int i = 0; i < H; i++) {
		Mat roi(*combined_img, roi_rect);
		tmp[i]->copyTo(roi);
		roi_rect.y += HEIGHT;
	}
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
	for (int i = 0; i < H;i++)delete tmp[i];
	return combined_img;
}

float operator-(const imgdata &obj1, const somap &obj2) {
	__m256 tmp;
	__m256 *v1 = (__m256*)(obj1.fvex);
	__m256 *v2 = (__m256*)(obj2.fvex);
	__m256 sum = _mm256_set1_ps(0);
	const __m256 signmask = _mm256_set1_ps(-0.0f); // 0x80000000
	for (int i = 0; i < f; i++) {
		tmp = _mm256_sub_ps(*(v1 + i), *(v2 + i));
		tmp = _mm256_andnot_ps(signmask, tmp);
		sum = _mm256_sub_ps(sum, tmp);
	}
	for (int i = 1; i < 8; i++)sum.m256_f32[0] += sum.m256_f32[i];
	if (isinf(sum.m256_f32[0]))return FLT_MAX;
	else if (isnan(sum.m256_f32[0]))return 0;
	return sum.m256_f32[0];
}