#include "som.h"

int somap::num = 0;
mt19937 somap::mt;

somap::somap()
{
	//cout << num << endl;
	this->x = this->num % W;
	this->y = this->num / H;
	this->fvex = (float*)_aligned_malloc(sizeof(float)*F256, 32);
	for (int i = 0; i < F; i++)this->fvex[i] = 0;
	static random_device rnd;
	this->mt.seed(rnd());
	this->num++;
}

somap::~somap()
{
	_aligned_free(this->fvex);
}

Mat* somap::picimg(imgdata *imgd) {
	vector<int> ilist;
	float min =FLT_MAX;//*this->setdata - *this;
	float* tmp = new float[HW];
	for (int i = 0; i < HW; i++)tmp[i] = (imgd[i] - *this);
	for (int i = 0; i < HW; i++){
		if (isinf(tmp[i]))tmp[i] = FLT_MAX;
		else if (isnan(tmp[i]))tmp[i] = 0;
		if (tmp[i] == min) {
			ilist.push_back(i);
		}else if(tmp[i] < min){
			min = tmp[i];
			ilist.clear();
			ilist.push_back(i);
		}
	}
	delete[] tmp;
	uniform_int_distribution<> randin(0, (ilist.size()-1));
	return imgd[ilist[randin(this->mt)]].img;
}

void somap::train(imgdata *imgd,imgdata *test,somap *smp,const int count) {
	int dist = abs(this->x-smp->x)+abs(this->y-smp->y);
	int vic=0;
	if (count < 500) { vic = 1; }
	else if (count<(1000)) { vic = HW*HW; }
	else if (count<(2000)) { vic = 1; }
	else if (count<(4000)) { vic = H; }
	else if (count<(6000)) { vic = H/2; }
	else if (count<(8000)) { vic = H; }
	else if (count<(10000)) { vic = H/4; }
	else if (count<(12000)) { vic = 20; }
	else if (count<(14000)) { vic = 10; }
	else if (count<(16000)) { vic = 5; }
	else if (count<(18000)) { vic = 3; }
	else if (count<(20000)) { vic = 2; }
	else { vic = 1; }
	if (dist > vic)return;
	float w = 0;
	if ((count <10000)&&(dist == 0)) { w=1; }
	else if (count<(500)) {w=(float)1 / ((float)dist * 1); }
	else if (count<(1000)) { w = (float)1 /  ((float)dist*50); }
	else if (count<(2000)) { w = (float)1 /  ((float)dist*5); }
	else if (count<(4000)) { w = (float)1 /  ((float)dist*100); }
	else if (count<(6000)) { w = (float)1 /  ((float)dist*1000); }
	else if (count<(8000)) { w = (float)1 / ((float)dist*2500); }
	else if (count<(10000)) { w = (float)1 / ((float)dist*5000); }
	else if (count<(12000)) { w = (float)1 / ((float)dist*10000); }
	else if (count<(14000)) { w = (float)1 /((float)dist*20000); }
	else if (count<(16000)) { w = (float)1 /((float)dist*30000); }
	else if (count<(18000)) { w =(float)1/((float)dist * 50000); }
	else if (count<(20000) && (dist == 0)) { w = 1 / ((float)1 * 10000); }
	else if (count<(20000)) { w = (float)1/((float)dist*40000); }
	else { w = 1 / ((float)dist* 1000000); }
	__m256 tmp[f];
	const __m256 ws = _mm256_broadcast_ss(&w);
	//const __m256 ws = _mm256_set1_ps(w);
	__m256 *v1 = (__m256*)(this->fvex);
	const __m256 *v2 = (__m256*)(test->fvex);
#ifdef _OPENMP
#pragma omp parallel
#endif
	{
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
		for (int i = 0; i < f; i++) tmp[i] = _mm256_sub_ps(*(v2 + i), *(v1 + i));
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
		for (int i = 0; i < f; i++)tmp[i] = _mm256_mul_ps(tmp[i], ws);
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
		for (int i = 0; i < f; i++)*(v1 + i) = _mm256_add_ps(*(v1 + i), tmp[i]);
	}
}

int somap::rand() {
	uniform_int_distribution<> randn(0, this->num - 1);
	return randn(this->mt);
}