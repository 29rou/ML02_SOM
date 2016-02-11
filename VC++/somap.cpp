#include "som.h"

int somap::num = 0;

somap::somap()
{
	//cout << num << endl;
	this->x = this->num % W;
	this->y = this->num / H;
	this->fvex = (float*)_aligned_malloc(sizeof(float)*F256, 32);
	for (int i = 0; i < F; i++)this->fvex[i] = 0;
	this->num++;
}


somap::~somap()
{
	_aligned_free(this->fvex);
}

void somap::picimg(imgdata *imgd) {
	random_device rnd;
	mt19937_64 mt(rnd());
	vector<int> is;
	float min = FLT_MAX;
	float tmp;
	for (int i = 0; i < HW; i++) {
		tmp = imgd[i] - *this;
		if (tmp <= min) {
			min = tmp;
			is.push_back(i);
		}
	}
	//for (auto &i : is)cout << i<<" ";
	if (is.empty())return;
	uniform_int_distribution<> randin(0, (is.size()-1));
	*this->img = *(imgd[is[randin(mt)]].img);
}

void somap::train(imgdata *imgd,imgdata *test,somap *smp,const int count) {
	int dist = abs(this->x-smp->x)+abs(this->y-smp->y);
	int vic=0;
	if (count == 0) { vic = HW; }
	else if (count<(10)) { vic = H; }
	else if (count<(2000)) { vic = 4; }
	else if (count<(4000)) { vic = 3; }
	else if (count<(8000)) { vic = 2; }
	else if (count<(10000)) { vic = 2; }
	else if (count<(32000)) { vic = 2; }
	else if (count<(100000)) { vic = 1; }
	else { vic = 1; }
	if (dist > vic)return;
	float w = 0;
	if (count == 0) { w = (1.0 / (float)dist); }
	else if (count<(1000)) { w = ((1.0 / (float)dist))/5.0; }
	else if (count<(2000)) { w = ((1.0 / (float)dist))/10.0; }
	else if (count<(4000)) { w = ((1.0 / (float)dist))/100.0; }
	else if (count<(8000)) { w = ((1.0 / (float)dist))/1000.0; }
	else if (count<(16000)) { w = ((1.0 / (float)dist))/10000.0; }
	else if (count<(32000)) { w = ((1.0 / (float)dist))/100000.0; }
	else if (count<(100000)) { w = ((1.0 / (float)dist))/1000000.0; }
	else { w = ((1.0 / (float)dist))/ 10000000.0; }
	__m256 tmp[f],tmp2[f];
	__m256 ws = _mm256_broadcast_ss(&w);
	__m256 *v1 = (__m256*)(this->fvex);
	__m256 *v2 = (__m256*)(test->fvex);
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int i = 0; i <f; i++) {
		tmp[i] = _mm256_sub_ps(*v2, *v1);
#pragma omp atomic
		v1++;
#pragma omp atomic
		v2++;
		tmp2[i] = _mm256_mul_ps(tmp[i], ws);
	}
	v1 = (__m256*)(this->fvex);
	v2 = (__m256*)(test->fvex);
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int i = 0; i < f; i++) {
		*v1 = _mm256_add_ps(*v1, tmp2[i]);
#pragma omp atomic
		v1++;
	}
	this->picimg(imgd);
}