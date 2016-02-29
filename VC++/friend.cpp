#include "som.h"

void initializemap(imgdatas &imgd, somaps &smp) {
	random_device rnd;
	mt19937_64 mt(rnd());
	uniform_int_distribution<> randc(0, imgd.size() - 1);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
	for (int i = 0; i < smp.size(); i++) {
		for (int j = 0; j < smp.data()->size(); j++) {
			smp.at(i).at(j)->init(imgd.at(randc(mt)));
		}
	}
}

void normalize(imgdatas &imgd){
	array<vector<float>,F> allvec;
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int i = 0; i < F; i++) {
		allvec.at(i).reserve(imgd.size());
	}
	array<map<float, float>,F> sample;
	auto load = [&allvec](imgdata &imgd) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int i = 0; i < F; i++) {
			allvec.at(i).push_back(imgd.fvex[i]);
		}
	};
	for_each(imgd.begin(), imgd.end(),load);
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int i = 0; i < allvec.size(); i++) {
		sort(allvec.at(i).begin(), allvec.at(i).end());
		auto result = unique(allvec.at(i).begin(), allvec.at(i).end());
		allvec.at(i).erase(result, allvec.at(i).end());
		cout << allvec.at(i).size() << "\n";
		for (int k = 0; k < allvec.at(i).size(); k++) {
			sample.at(i).insert(make_pair(allvec.at(i).at(k), (k + 1) * 10));
		}
		allvec.at(i).clear();
		allvec.at(i).shrink_to_fit();
	}
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int i = 0; i < imgd.size();i++) {
		for (int j = 0; j < F; j++) {
			imgd.at(i).fvex[j] = sample.at(j).find(imgd.at(i).fvex[j])->second;
		}
	}
}

void showimg(combinedimg &cmb) {
	namedWindow("SOMing", CV_WINDOW_AUTOSIZE);
	Mat img;
	resize(cmb.cmbimg, img, Size(W*WIDTH / 3, H*HEIGHT / 3));
	imshow("SOMing", img);
	waitKey(0);
}
