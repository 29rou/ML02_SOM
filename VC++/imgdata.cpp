#include "som.h"

int imgdata::num = 0;

imgdata::imgdata()
{
	this->fvex = (float*)_aligned_malloc(sizeof(float)*F256, 32);
	for (int i = 0; i < F;i++)this->fvex[i] = 0;
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
	resize(src_img, src_img, Size(96, 96));
	Mat img;
	cvtColor(src_img, img, CV_RGB2GRAY);
	HOGDescriptor hog(Size(96, 96), Size(32, 32), Size(16, 16), Size(16, 16), 9, -1, 0.2, true, 64);
	vector<Point> locations;
	vector<float> featureVec;
	hog.compute(img, featureVec, Size(0, 0), Size(0, 0), locations);
	resize(src_img, src_img, Size(WIDTH, HEIGHT), CV_INTER_CUBIC);
	//cvtColor(src_img, src_img, CV_RGB2GRAY, CV_INTER_CUBIC);
	this->img = new Mat(Size(WIDTH, HEIGHT), CV_8UC3);
	*this->img = src_img;
	for (int i = 0; i < F; i++)this->fvex[i] = featureVec[i];
}
