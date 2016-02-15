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
			BOOL bFinding2 = find2.FindFile("*.jpg");
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
	omp_set_num_threads(8);
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

void mkimg(imgdata* imgd, somap* smp, Mat* cmbimg) {
	cmbimg = toimg(imgd,smp, cmbimg);
	imshow("SOMing", *cmbimg);
	waitKey(0);
}

void outputimg(imgdata* imgd, somap* smp, const int count,Mat* cmbimg) {
	cmbimg = toimg(imgd,smp, cmbimg);
	string outputstr = ".\\output\\output" + to_string(count) + ".jpg";
	imwrite(outputstr, *cmbimg);
	waitKey(0);
}

Mat* som(imgdata *imgd, somap *smp) {
	somap* ptr;
	imgdata* test;
	Mat* cmbimg = new Mat(Size(W/3*WIDTH, H/3*HEIGHT), CV_8UC3);
	//cmbimg = toimg(imgd,smp, cmbimg);
	int num = imgd->num;
	int *count = new int;
	*count = 0;
	uniform_int_distribution<> randc(0, imgd->num - 1);	
	for (int n = 0;;n++) {
		//thread t1(mkimg, ref(imgd), ref(smp), ref(cmbimg));
		random_device rnd;
		mt19937_64 mt(rnd());
		//if (n % 10 == 0) {
			thread t2(outputimg, ref(imgd), ref(smp), ref(*count), ref(cmbimg));
			t2.join();
		//}
		for (int i = 0; i < 10000; i++) {
			++*count;
			printf("%020d\n", *count);
			test = imgd + randc(mt);
			ptr = test->findnear(smp);
			for (int j = 0; j < HW; j++) {
				(smp + j)->train(imgd, test, ptr, *count);
			}
		}
		//t2.detach();
	}
	delete count;
	for (int i = 0; i < HW; i++)smp[i].picimg(imgd);
	cmbimg = toimg(imgd,smp, cmbimg);
	return cmbimg;
}

void main() {
	string path = ".\\256_ObjectCategories\\";
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