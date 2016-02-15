#pragma once
#include <afx.h>
#include <omp.h>
#include <immintrin.h>
#include <malloc.h>
#include <iostream>
#include <random>
#include <thread>
#include <afxsock.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/objdetect/objdetect.hpp>

using namespace std;
using namespace cv;

constexpr int  N = 10000 * 100;
constexpr int  F = 65025;
constexpr int F256 = (F / 8 + 1)*8;
constexpr int f = F256 / 8;
constexpr int f2 = (f / 8 + 1) * 8;
constexpr int P = 8;
constexpr int H = 99;
constexpr int W = 99;
constexpr int HW = H*W;
constexpr int HEIGHT = 100;
constexpr int WIDTH = 100;

class somap;

class imgdata
{
	friend somap* initialize(imgdata *imgd);
	friend float operator-(const imgdata &obj1,const somap &obj2);
public:
	imgdata();
	~imgdata();
	Mat *img;
	float *fvex;
	int rand();
	void loadimg(const string &filepath);
	somap* findnear(somap *smp);
	static int num;
private:
	
	static mt19937 mt;
};

class somap
{
	friend somap* initialize(imgdata *imgd);
	friend Mat* toimg(imgdata* imgd,somap *smp, Mat* combined_img);
	friend float operator-(const imgdata &obj1,const somap &obj2);
public:
	somap();
	~somap();
	Mat* picimg(imgdata *imgd);
	int rand();
	void train(imgdata *imgd, imgdata *test, somap* smp,const int count);
private:
	static int num;
	static mt19937 mt;
	int x, y;
	float *fvex;
	//imgdata *setdata;
};