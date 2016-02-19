#include "som.h"

int imgdata::num = 0;
mt19937 imgdata::mt;

imgdata::imgdata()
{
	this->fvex = (float*)_aligned_malloc(sizeof(float)*F256, 32);
	//__m256 *v = (__m256*)(this->fvex);
	//for (int i = 0; i < f;i++)v[i]= _mm256_set1_ps(0);
	static random_device rnd;
	this->mt.seed(rnd());
	this->num++;
}

imgdata::~imgdata()
{
	_aligned_free(this->fvex);
	delete this->img;
}

void imgdata::loadimg(const string &filepath) {
	//cout << filepath;
	Mat src_img = imread(filepath);
	Mat img;
	vector<Point> locations;
	vector<float> featureVec;
	cv::resize(src_img, src_img, Size(120, 120));
	cvtColor(src_img, img, CV_RGB2GRAY);
	HOGDescriptor hog(Size(120, 120), Size(30,30), Size(6, 6), Size(15, 15), 9, -1, 0.2, true, 64);
	hog.compute(img, featureVec, Size(0, 0), Size(0, 0), locations);
	for (int i = 0; i < F; i++) {
		if (isinf(featureVec[i]))this->fvex[i] = FLT_MAX;
		else if (isnan(featureVec[i]))this->fvex[i] = 0;
		else this->fvex[i] = featureVec[i];
	}
	resize(src_img, src_img, Size(WIDTH, HEIGHT), CV_INTER_CUBIC);
	this->img = new Mat(Size(WIDTH, HEIGHT), CV_8UC3);
	*this->img = src_img;
}

somap* imgdata::findnear(somap *smp) {
	float min = FLT_MAX;
	//vector<int> ilist;
	int is;
	float *tmp = new float[HW];
	//float tmp[HW];
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
	for (int i = 0; i < HW; i++)tmp[i] = *this - smp[i];
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int i = 0; i < HW; i++){
		if (tmp[i] < min) {
#pragma omp critical
		{
			min = tmp[i];
			is = i;
		}
		}
	}
	delete[] tmp;
	//uniform_int_distribution<> randin(0, (ilist.size() - 1));
	//return smp+ilist[randin(this->mt)];
	return smp+is;
}

int imgdata::rand() {	
	uniform_int_distribution<> randn(0, this->num - 1);
	return randn(this->mt);
}