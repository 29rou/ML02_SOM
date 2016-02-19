#include "som.h"

 vector<string>* find_file(const string &path) {
	vector<string> *filepath = new vector<string>;
	CFileFind find;
	char tmpCurrentDir[MAX_PATH];
	GetCurrentDirectory(sizeof(tmpCurrentDir), tmpCurrentDir);
	cout << tmpCurrentDir << endl;
	SetCurrentDirectory(path.c_str());
	BOOL bFinding = find.FindFile();
	while (bFinding) {
		bFinding = find.FindNextFile();
		if (find.IsDirectory()) {
			CFileFind find2;
			string tmpath = find.GetFilePath();
			SetCurrentDirectory(tmpath.c_str());
			BOOL bFinding2 = find2.FindFile("*.png");
			while (bFinding2) {
				bFinding2 = find2.FindNextFile();
				if (find2.IsDirectory()==0) {
					(*filepath).push_back((string)find2.GetFilePath());
					//printf("%s\n", find2.GetFilePath());
				}
			}
			SetCurrentDirectory(path.c_str());
		}
	}
	SetCurrentDirectory(tmpCurrentDir);
	return filepath;
};

imgdata* img_tovec(vector<string> *filepath) {
	const int filenum = (*filepath).size();
	imgdata *ptr = new imgdata[filenum];
	vector<int> chk;
	random_device rnd;
	uniform_int_distribution<> randf(0, filenum-1);
	int *tmp = new int[filenum];
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
	for (int i = 0; i < filenum; i++) {
			do {
				tmp[i] = randf(rnd);
			} while (find(chk.begin(), chk.end(), tmp[i]) != chk.end());
#pragma omp critical
			chk.push_back(tmp[i]);
			//cout<<"load "<<(*filepath)[tmp[i]]<<"\n";
		ptr[i].loadimg((*filepath)[tmp[i]]);
	}
	delete[] tmp;
	return ptr;
}

imgdata* load_img(const string &path) {
	vector<string> *filepath = find_file(path);
	imgdata *ptr = img_tovec(filepath);
	(*filepath).clear();
	(*filepath).shrink_to_fit();
	delete filepath;
	return ptr;
}

void mkimg(Mat* cmbimg) {
	imshow("SOMing", *cmbimg);
	waitKey(0);
}

void outputimg(imgdata* imgd, somap* smp, const int count,Mat* cmbimg) {
	cmbimg = toimg(imgd,smp, cmbimg);
	string outputstr = ".\\output\\output" + to_string(count) + ".jpg";
	imwrite(outputstr, *cmbimg);
	//imshow("SOMing", *cmbimg);
	waitKey(1);
}

int setvic(const int &count) {
	int vic = 1;
	if (count<(10)) { vic += HW; }
	else if (count<(10000)) { vic += H; }
	else if (count<(20000)) { vic += H/2; }
	else if (count<(40000)) { vic += H/2; }
	else if (count<(80000)) { vic += H/2; }
	else if (count<(160000)) { vic += 5; }
	else if (count<(320000)) { vic += 4; }
	else if (count<(640000)) { vic += 1; }
	else { vic += 3; }
	return vic;
}

Mat* som(imgdata *imgd, somap *smp) {
	somap* ptr;
	imgdata* test;
	Mat* cmbimg = new Mat(Size(W*WIDTH, H*HEIGHT), CV_8UC3);
	const int num = imgd->num;
	random_device rnd;
	normal_distribution<>randc((num - 1) / 2, 500.0);
	int vic = 1;
	for (int count =0;;) {
		mt19937_64 mt(rnd());
		vic = setvic(count);
		if(count%10000==0)outputimg(imgd, smp, count, cmbimg);
		for (int i= 0; i < 1000; i++) {
			//outputimg(imgd, smp, count, cmbimg);
			printf("%020d\n", count);
			test = imgd+(int)randc(mt);
			ptr = test->findnear(smp);
#ifdef _OPENMP
#pragma omp parallel for
#endif
			for (int j = 0; j < HW; j++)(smp + j)->train(imgd, test, ptr, count,&vic);
			++count;
		}
	}
	cmbimg = toimg(imgd,smp, cmbimg);
	return cmbimg;
}

void main() {
	string path = ".\\thumb\\";
	imgdata *imgd = load_img(path);
	//cout << "finish load" << endl;
	somap *smp = initialize(imgd);
	//cout << "Start!!" << endl;
	Mat* cmbimg = som(imgd, smp);
	delete[] smp;
	delete[] imgd;
	imshow("SOM", *cmbimg);
	waitKey(0);
	delete cmbimg;
}