#include "som.h"

int imgdata::num = 0;
mt19937 imgdata::mt;

imgdata::imgdata()
{
	this->fvex = (float*)_aligned_malloc(sizeof(float)*F256, 32);
	for (int i = 0; i < F;i++)this->fvex[i] = 0;
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
	resize(src_img, src_img, Size(100, 100));
	cvtColor(src_img, img, CV_RGB2GRAY);
	HOGDescriptor hog(Size(100, 100), Size(20, 20), Size(5, 5), Size(4, 4), 9, -1, 0.2, true, 64);
	hog.compute(img, featureVec, Size(0, 0), Size(0, 0), locations);
	if (featureVec.empty()) {
		cout << "ERROR" << endl;
	}
	for (int i = 0; i < F; i++) {
		if (isinf(featureVec[i])) {
			this->fvex[i] = FLT_MAX;
		}
		else if (isnan(featureVec[i])) {
			this->fvex[i] = 0;
		}
		else this->fvex[i] = featureVec[i];
	}
	resize(src_img, src_img, Size(WIDTH, HEIGHT), CV_INTER_CUBIC);
	this->img = new Mat(Size(WIDTH, HEIGHT), CV_8UC3);
	*this->img = src_img;
}

somap* imgdata::findnear(somap *smp) {
	float min = FLT_MAX;
	vector<int> is;
	float* tmp = new float[HW];
	for (int i = 0; i < HW; i++)tmp[i] = *this - smp[i];
	for (int i = 0; i < HW; i++){
		if (tmp[i] == min) {
			is.push_back(i);
		}
		else if (tmp[i] < min) {
			min = tmp[i];
			is.clear();
			is.push_back(i);
		}
	}
	delete[] tmp;
	uniform_int_distribution<> randin(0, (is.size() - 1));
	return smp + is[randin(this->mt)];
}

int imgdata::rand() {	
	uniform_int_distribution<> randn(0, this->num - 1);
	return randn(this->mt);
}