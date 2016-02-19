#include "som.h"

int somap::num = 0;
mt19937 somap::mt;

somap::somap()
{
	//cout << num << endl;
	this->x = this->num % W;
	this->y = this->num / H;
	this->fvex = (float*)_aligned_malloc(sizeof(float)*F256, 32);
	//__m256 *v = (__m256*)(this->fvex);
	//for (int i = 0; i < f; i++)v[i] = _mm256_set1_ps(0);
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
	float tmp[HW];
//#ifdef _OPENMP
//#pragma omp parallel for schedule(static)
//#endif
	for (int i = 0; i < HW; i++) {
		tmp[i] = (imgd[i] - *this);
	}
//#ifdef _OPENMP
//#pragma omp parallel for
//#endif
	for (int i = 0; i < HW; i++){
		if (tmp[i] == min) {
//#pragma omp critical
			ilist.push_back(i);
		}else if(tmp[i] < min){
//#pragma omp critical
		{
			min = tmp[i];
			ilist.clear();
			ilist.push_back(i);
		}
		}
	}
	uniform_int_distribution<> randin(0, (ilist.size()-1));
	return imgd[ilist[randin(this->mt)]].img;
}

void somap::train(imgdata *imgd,imgdata *test,somap *smp,const int count, const int *vic) {
	int dist = abs(this->x - smp->x) + abs(this->y - smp->y)+1;
	if (dist > *vic)return;
	float w = 0;
	if (count<(10)) { w = 1; }
	else if (count<(1000))   { w = 1; }
	else if (count<(5000))   { w = 0.9; }
	else if (count<(20000))  { w = 0.9; }
	else if (count<(40000))  { w = 0.9; }
	else if (count<(840000)) { w = 0.9; }
	else if (count<(160000)) { w = 0.009; }
	else if (count<(320000)) { w = 0.005; }
	else if (count<(640000)) { w = 0.002; }
	else if (dist == 1) { w = 0.00001; }
	else if (dist == 2) { w = 0.000009; }
	else if (dist == 3){ w = 0.000005; }
	else { w = 0.0005; }
	__m256 tmp;
	const __m256 ws = _mm256_broadcast_ss(&w);
	__m256 *v1 = (__m256*)(this->fvex);
	const __m256 *v2 = (__m256*)(test->fvex);
	for (int i = 0; i < f; i++) {
		tmp = _mm256_sub_ps(*(v2 + i), *(v1 + i));
		tmp = _mm256_mul_ps(tmp, ws);
		*(v1 + i) = _mm256_add_ps(*(v1 + i), tmp);
	}
}

int somap::rand() {
	uniform_int_distribution<> randn(0, this->num - 1);
	return randn(this->mt);
}