#pragma once
#include <afx.h>
#include <immintrin.h>
#include <array>  
#include <map>
#include <numeric>
#include <algorithm> 
#include <iostream>
#include <random>
#include <thread>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

using namespace std;
using namespace cv;

constexpr int  F = 1296;
constexpr int F256 = (F / 8 + 1) * 8;
constexpr int f = F256 / 8;
constexpr int H = 50;
constexpr int W = 50;
constexpr int HW = H*W;
constexpr int HEIGHT = 180;
constexpr int WIDTH = 320;

class somap;
class imgdata;
using imgdatas = vector<imgdata>;
using somaps = array<array<somap*, W>, H>;

class sombase
{
public:
	sombase();
	~sombase();
private:
	static std::random_device rnd;
protected:
	static std::mt19937_64 mt;
	float* fvex;
	const float getDistance(const sombase& obj);
};

class combinedimg
{
	friend void showimg(combinedimg &cmb);
public:
	combinedimg();
	combinedimg(imgdatas &imgd, somaps &smp);
	~combinedimg();
	void outputimg(const int count);
	void toimg(imgdatas &imgd, somaps &smp);
	const Mat* showimg() { return  &this->cmbimg; }
private:
	static CTime theTime;
	static string time;
	Mat cmbimg;
};

class somap :
	public sombase
{
	friend void combinedimg::toimg(imgdatas &imgd, somaps &smp);
	friend void initializemap(imgdatas &imgd, somaps &smp);
public:
	somap();
	~somap();
	void setw(const int &count, const pair<int,int> &ptr, const int &x,const int &y);
	void train(const imgdata & obj);
private:
	float weight;
	void init(imgdata &imgd);
	void getnearlist(imgdatas & imgdata, vector<Mat*> &matlist);
};

class imgdata :
	public sombase
{
	friend somap;
	friend void initializemap(imgdatas &imgd, somaps &smp);
	friend void normalize(imgdatas &imgd);
public:
	imgdata();
	~imgdata();
	void loadimg(const string &filepath);
	const pair<int, int> findnear(somaps &smp);
private:
	Mat img;
};
