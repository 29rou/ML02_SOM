#include "som.h"

Mat* toimg(somap *smp, Mat* combined_img);

 vector<string>* find_file(const string &path) {
	vector<string> *filepath = new vector<string>;
	CFileFind find;
	char tmpCurrentDir[MAX_PATH];
	GetCurrentDirectory(sizeof(tmpCurrentDir), tmpCurrentDir);
	SetCurrentDirectory(path.c_str());
	BOOL bFinding = find.FindFile();
	char* jpg = "jpg";
	while (bFinding) {
		bFinding = find.FindNextFile();
		if (find.IsDirectory()) {
			CFileFind find2;
			string tmpath = find.GetFilePath();
			SetCurrentDirectory(tmpath.c_str());
			BOOL bFinding2 = find2.FindFile();
			while (bFinding2) {
				bFinding2 = find2.FindNextFile();
				if (find2.IsDirectory()==0) {
					(*filepath).push_back((string)find2.GetFilePath());
				}
			}
			SetCurrentDirectory(path.c_str());
		}
		/*else if(find.IsDots()){}
		else if ((find.GetFileName()).ReverseFind(*jpg)>=0) {
			(*filepath).push_back((string)find.GetFilePath());
		}*/
	}
	return filepath;
};

imgdata* img_tovec(vector<string> *filepath) {
	const int filenum = (*filepath).size();
	imgdata *ptr = new imgdata[filenum];
	//cout << "check" << endl;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
	for (int i = 0; i < filenum; i++)ptr[i].loadimg((*filepath)[i]);
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

somap* initialize(imgdata *imgd) {
	random_device rnd;
	mt19937_64 mt(rnd());
	somap *ptr = new somap[HW];
	vector<int> chk;
	uniform_int_distribution<> randV(0, imgd->num-1);
	int tmp;
	for (int i = 0; i < HW; i++) {
		do{
			tmp = randV(mt);
		} while (find(chk.begin(), chk.end(), tmp) != chk.end());
		chk.push_back(tmp);
		initialize(&(imgd[tmp]), &(ptr[i]));
		//ptr[i].picimg(imgd);
		//cout << tmp << " "<< (ptr[i]).img->cols << endl;
	}
	//imshow("som", *(ptr[3].img));
	//waitKey(0);
	return ptr;
}

void compute(imgdata *imgd,somap *smp) {
	random_device rnd;
	mt19937_64 mt(rnd());
	somap* ptr;
	imgdata* test;
	uniform_int_distribution<> randc(0, imgd->num-1);
	for (int i = 0; i < N; i++) {
		test = imgd + randc(mt);
		ptr = findnear(test, smp);
		if (ptr == 0)continue;
		for (int j = 0; j < HW; j++) {
			(smp + j)->train(imgd,test,ptr, i);
		}
	}
}

void showimg(somap *smp,Mat*cmbimg) {
	while(1){
		//cout<<n<<"/"<<max<<endl;
		Mat* ptr=toimg(smp,cmbimg);
		imshow("SOM", *ptr);
		waitKey(1);
		//cout << "ok" << endl;
		//chrono::milliseconds waittime(998);
		//this_thread::sleep_for(chrono::milliseconds(waittime));
	}
}

void sckt(Mat* cmbimg) {
	/*CSocket sock;
	unsigned int port;
	sock.Create();
	sock.Connect(port);
	sock.Listen(1);*/
	while (1) {
		//cout<<n<<"/"<<max<<endl;
		imshow("SOM", *cmbimg);
		waitKey(1);
		//cout << "ok" << endl;
		chrono::milliseconds waittime(998);
		this_thread::sleep_for(chrono::milliseconds(waittime));
	}
}

void som(imgdata *imgd, somap *smp,Mat*cmbimg) {
	int n = 0;
	//cout<<flush<<c<<" "<<n<<"/"<<max<<endl;
	thread t1(compute, ref(imgd),ref(smp));
	thread t2(showimg, ref(smp),ref(cmbimg));
	//thread t3(sckt, ref(cmbimg));
	t1.join();
	t2.join();
	//t3.join();
}

Mat* toimg(somap *smp,Mat* combined_img) {
	Rect roi_rect;
	roi_rect.width = WIDTH;
	roi_rect.height = HEIGHT;
	for (int i = 0; i <HW; i++) {
		Mat roi(*combined_img, roi_rect);
		(*(smp[i].img)).copyTo(roi);
		roi_rect.x += WIDTH;
		if (smp[i].x == W - 1) {
			roi_rect.x = 0;
			roi_rect.y += HEIGHT;
		}
	}
	//imshow("SOM", combined_img);
	return combined_img;
}

void main() {
	string path = "C:\\Users\\iikun\\Desktop\\新しいフォルダー (2)\\256_ObjectCategories\\";
	imgdata *imgd = load_img(path);
	Mat* cmbimg = new Mat(Size(W*WIDTH, H*HEIGHT), CV_8UC3);
	cout << "finish load" << endl;
	somap *smp = initialize(imgd);
	som(imgd, smp,cmbimg);
	toimg(smp,cmbimg);
	imshow("SOM", *cmbimg);
	waitKey(0);
	delete imgd;
	delete[] smp;
	delete[] imgd;
	cin >> path;
}
