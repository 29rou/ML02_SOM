#include "som.h"

void initializemap(imgdatas &imgd, somaps &smp) {
	std::random_device rnd;
	std::mt19937_64 mt(rnd());
	std::uniform_int_distribution<> randc(0, imgd.size() - 1);
	std::vector<imgdata*> matlist;
	std::vector<std::pair<int, int>> xy;
	somap *central = smp.at(smp.size() / 2).at(smp.data()->size() / 2);
	central->init(imgd.at(randc(mt)));
	central->getnearlist(imgd, matlist);
	auto forsort = [&smp](std::pair<int, int> &x, std::pair<int, int> &y) -> bool {
		auto dist = [&smp](std::pair<int, int> &x) ->int {
			return std::abs(smp.data()->size() / 2 - x.first) + std::abs(smp.size()/2 - x.second); 
		};
		if (dist(x) > dist(y))return false;
		if (dist(x) < dist(y))return true;
		return false;
	};
	for (int i = 0; i < smp.size(); i++) {
		for (int j = 0; j < smp.data()->size(); j++) {
			xy.push_back({ j,i });
		}
	}
	sort(xy.begin(), xy.end(),forsort);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
	for (int i = 0; i < smp.size()*smp.data()->size();i++) {
		smp.at(xy.at(i).second).at(xy.at(i).first)->init(*matlist.at(i));
	}
}

void normalize(imgdatas &imgd){
	std::array<std::vector<float>,F> allvec;
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int i = 0; i < F; i++) {
		allvec.at(i).reserve(imgd.size());
	}
	std::array<std::map<float, float>,F> sample;
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
		printf("%d\n", allvec.at(i).size());
		for (int k = 0; k < allvec.at(i).size(); k++) {
			sample.at(i).insert(std::make_pair(allvec.at(i).at(k), (k + 1)* 10));
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
