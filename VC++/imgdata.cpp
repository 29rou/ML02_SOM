#include "som.h"

imgdata::imgdata()
{
}

imgdata::~imgdata()
{
}

void imgdata::loadimg(const std::string & filepath)
{
	Mat src_img = imread(filepath);
	normalize(src_img, src_img, 0, 255, NORM_MINMAX);
	//cout << filepath << "\n";
	Mat img;
	std::vector<Point> locations;
	std::vector<float> featureVec;
	this->img = Mat::zeros(Size(WIDTH, HEIGHT), CV_8UC3);
	HOGDescriptor hog(Size(640, 360), Size(160, 120), Size(160, 120), Size(40, 40), 9, -1, 0.2, true, 64);
	resize(src_img, img, Size(640, 360), CV_INTER_CUBIC);
	hog.compute(img, featureVec, Size(0, 0), Size(0, 0), locations);
	for (int i = 0; i < featureVec.size(); i++)this->fvex[i] = featureVec.at(i);
	resize(src_img, src_img, Size(WIDTH, HEIGHT), CV_INTER_CUBIC);
	this->img = src_img;
}

const std::pair<int,int> imgdata::findnear(somaps &smp) {
	using cordinate = std::pair<int, int>;
	std::vector<cordinate> ilist;
	std::array<std::pair<cordinate,float>, HW> tmp;
	auto forsort = [](std::pair< cordinate, float> &x, std::pair< cordinate, float> &y) -> bool {
		if (x.second > y.second)return false; 
		if (x.second < y.second)return true; 
		return false; 
	};
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
	for (int i = 0; i < smp.size(); i++) {
		for (int j = 0; j < smp.data()->size(); j++) {
			tmp.at(i*W + j) = { {j,i},this->getDistance(*smp.at(i).at(j)) };
		}
	}
	std::sort(tmp.begin(),tmp.end(), forsort);
	for (auto &i:tmp) {
		if (tmp.front().second != i.second)break;
		ilist.push_back(i.first);
	}
	std::uniform_int_distribution<> randin(0, (ilist.size() - 1));
	return ilist.at(randin(this->mt));
}
