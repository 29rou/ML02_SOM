#include "som.h"

void find_file(const std::string &path ,std::vector<std::string> &filepath) {
	CFileFind find;
	char tmpCurrentDir[MAX_PATH];
	GetCurrentDirectory(sizeof(tmpCurrentDir), tmpCurrentDir);
	std::cout << tmpCurrentDir << std::endl;
	SetCurrentDirectory(path.c_str());
	BOOL bFinding = find.FindFile();
	while (bFinding) {
		bFinding = find.FindNextFile();
		if (find.IsDirectory()) {
			CFileFind find2;
			std::string tmpath = find.GetFilePath();
			SetCurrentDirectory(tmpath.c_str());
			BOOL bFinding2 = find2.FindFile("*.jpg");
			while (bFinding2) {
				bFinding2 = find2.FindNextFile();
				if (find2.IsDirectory() == 0) {
					filepath.push_back((std::string)find2.GetFilePath());
					//printf("%s\n", find2.GetFilePath());
				}
			}
			SetCurrentDirectory(path.c_str());
		}
	}
	SetCurrentDirectory(tmpCurrentDir);
}

void img_tovec(std::vector<std::string> &filepath, imgdatas &imgd) {
	std::random_device rnd;
	std::mt19937_64 mt(rnd());
	shuffle(filepath.begin(), filepath.end(), mt);
	imgd.reserve(filepath.size());
	for (int i = 0; i < filepath.size(); i++) {
		imgd.emplace_back();
	}
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
	for (int i = 0; i < filepath.size();i++) {
		imgd.at(i).loadimg(filepath.at(i));
	}
}

void som(imgdatas *imgd, somaps &smp) {
	std::pair<int,int> ptr;
	imgdata* test;
	combinedimg cmb(*imgd, smp);
	std::random_device rnd;
	std::mt19937_64 mt(rnd());
	std::uniform_int_distribution<> randc(0, imgd->size() - 1);
	for (int count = 0;;) {
		if (count % 100000 == 0) {
			cmb.toimg(*imgd, smp);
			cmb.outputimg(count);
		}
		//cmb.outputimg(count);
		//thread t1(showimg, ref(cmb));
		printf("%010d\n", count);
		for (int i = 0; i < 10000; i++,count++) {
			test = &imgd->at(randc(mt));
			ptr = test->findnear(smp);
#ifdef _OPENMP
#pragma omp parallel for
#endif
			for (int j = 0; j < H; j++) {
				for (int k = 0; k < W; k++) {
					smp.at(j).at(k)->setw(count, ptr,k, j);
					smp.at(j).at(k)->train(*test);
				}
			}
		}
		//cmb.toimg(*imgd, smp);
		//t1.detach();
	}	
}

void main() {
	somap *ptr = new somap[HW];
	somaps smp;
	for (int i = 0; i < smp.size(); i++) {
		for (int j = 0; j < smp.data()->size(); j++) {
			smp.at(i).at(j) = &ptr[i*W + j];
		}
	}
	std::vector<std::string> filepath;
	find_file(".\\gusa\\",filepath);
	imgdatas imgd;
	img_tovec(filepath, imgd);
	filepath.clear();
	filepath.shrink_to_fit();
	normalize(imgd);
	initializemap(imgd, smp);
	som(&imgd, smp);
	delete[] ptr;
}