#include "som.h"

CTime combinedimg::theTime = CTime::GetCurrentTime();
std::string combinedimg:: time  = ".\\output\\" + theTime.Format(_T("%d%H%M"));

combinedimg::combinedimg()
{
	this->cmbimg = Mat::zeros(Size(W*WIDTH, H*HEIGHT), CV_8UC3);
	CreateDirectory(this->time.c_str(), NULL);
}

combinedimg::combinedimg(imgdatas & imgd, somaps & smp)
{
	this->cmbimg = Mat::zeros(Size(W*WIDTH, H*HEIGHT), CV_8UC3);
	CreateDirectory(this->time.c_str(), NULL);
	this->toimg(imgd, smp);
}


combinedimg::~combinedimg()
{
}

void combinedimg::outputimg(const int count)
{
	std::string str = time + "\\output" + std::to_string(count) + ".jpg";
	printf("%s\n", str.c_str());
	imwrite(str, this->cmbimg);
	waitKey(1);
}

void combinedimg::toimg(imgdatas & imgd, somaps & smp)
{
	Rect roi_rect;
	roi_rect.width = W*WIDTH;
	roi_rect.height = HEIGHT;
	std::array<std::array<Mat*, W>,H> img;
	std::array<std::array<std::vector<Mat*>, W>, H> distlist;
	std::vector<Mat*> used;
	used.reserve(HW);
	std::array<Mat, H> tmp;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
	for (int i = 0; i < smp.size(); i++) {
		for (int j = 0; j < smp.data()->size(); j++) {
			smp.at(i).at(j)->getnearlist(imgd,distlist.at(i).at(j));
		}
	}
	for (int i = 0; i < smp.size(); i++) {
		for (int j = 0; j < smp.data()->size(); j++) {
			for (auto &k:distlist.at(i).at(j)) {
				if (find(used.begin(), used.end(), k) == used.end()) {
					img.at(i).at(j) = k;
					used.push_back(k);
					break;
				}
			}
		}
	}
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
	for (int i = 0; i <smp.size(); i++) {
		tmp.at(i) = Mat::zeros(Size(W*WIDTH, HEIGHT), CV_8UC3);
		Rect roi_tmp;
		roi_tmp.width = WIDTH;
		roi_tmp.height = HEIGHT;
		for (int j = 0; j < smp.data()->size(); j++) {
			Mat roi(tmp.at(i), roi_tmp);
			img.at(i).at(j)->copyTo(roi);
			roi_tmp.x += WIDTH;
		}
	}
	for (auto &i : tmp) {
		Mat roi(this->cmbimg, roi_rect);
		i.copyTo(roi);
		roi_rect.y += HEIGHT;
	}
	return;
}
