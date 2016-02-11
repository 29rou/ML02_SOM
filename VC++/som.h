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

#define N  10000*100
#define F  900
constexpr int F256 = (900 / 8 + 1)*8;
constexpr int f = F256 / 8;
#define P  8
#define H 20
#define W 20
constexpr int HW = H*W;
#define HEIGHT 40
#define WIDTH 40


class somap;

class imgdata
{
	friend void initialize(imgdata *src, somap *dst);
	friend float operator-(const imgdata &obj1,const somap &obj2);
	friend somap* findnear(const imgdata *imgd, somap *smp);
public:
	imgdata();
	~imgdata();
	static int num;
	void loadimg(const string &filepath);
	Mat *img;
	float *fvex;
private:
	
};

class somap
{
	friend void initialize(imgdata *src, somap *dst);
	friend float operator-(const imgdata &obj1,const somap &obj2);
	friend somap* findnear(const imgdata *imgd, somap *smp);
public:
	somap();
	~somap();
	Mat *img;
	int x, y;
	void picimg(imgdata *imgd);
	void train(imgdata *imgd, imgdata *test, somap* smp,const int count);
private:
	static int num;
	float *fvex;
};



