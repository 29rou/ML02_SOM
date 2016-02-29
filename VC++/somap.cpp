#include "som.h"


somap::somap()
{
}


somap::~somap()
{
}

void somap::init(imgdata &imgd)
{
	for (int k = 0; k < F; k++) {
		this->fvex[k] = imgd.fvex[k];
	}
}

void somap::setw(const int &count, const pair<int,int> &ptr, const int &x,const int &y) {
	if (count < 100000) {
		int dist = abs(y - ptr.second) + abs(x - ptr.first);
		if (count < 1000 && dist < H*1.5)this->weight = 1.0;
		else if (count < 10000 && dist < H)this->weight = 1.0;
		else if (count < 50000 && dist < H / 2)this->weight = 0.9;
		else if (dist < H / 4)this->weight = 0.5;
		//else if (count < 100000 && dist < H / 5)this->weight = 0.0001;
		else this->weight = -1;
	}
	else {
		switch (abs(y - ptr.second) + abs(x - ptr.first)) {
		case 0:
			this->weight = 0.001;
			break;
		case 1:
		case 2:
			this->weight = 0.0009;
			break;
		case 3:
		case 4:
			this->weight = 0.0005;
			break;
		case 5:
		case 6:
		case 7:
		case 8:
		case 9:
		case 10:
			this->weight = 0.0003;
			break;
		case 11:
		case 12:
		case 13:
		case 14:
		case 15:
			this->weight = 0.0001;
			break;
		default:
			this->weight = -1;
			break;
		}
	}
}

void somap::train(const imgdata & obj)
{
	if (this->weight <= 0.0)return;
	__m256 tmp;
	__m256 *v1 = (__m256*)(this->fvex);
	const __m256 *v2 = (__m256*)(obj.fvex);
	const __m256 ws = _mm256_set1_ps(this->weight);
	for (int i = 0; i < f; i++) {
		tmp = _mm256_sub_ps(v2[i], v1[i]);
		tmp = _mm256_mul_ps(tmp, ws);
		v1[i] = _mm256_add_ps(v1[i], tmp);
	}
}

void somap::getnearlist(imgdatas & imgdata,vector<Mat*> &matlist)
{
	vector<pair<int, float>> tmp;
	tmp.reserve(imgdata.size());
	auto forsort = [](pair<int, float> &x, pair<int, float> &y) -> bool {
		if (x.second > y.second)return false;
		if (x.second < y.second)return true;
		return false;
	};
	auto forunique = [](pair<int, float> &x, pair<int, float> &y) -> bool {
		if (x.second == y.second)return true;
		return false;
	};
	for (int i = 0; i < imgdata.size(); i++)tmp.push_back({ i,this->getDistance(imgdata.at(i)) });
	sort(tmp.begin(), tmp.end(), forsort);
	auto result = unique(tmp.begin(), tmp.end(), forunique);
	tmp.erase(result, tmp.end());
	matlist.reserve(imgdata.size());
	for (auto &i : tmp) {
		matlist.push_back(&imgdata.at(i.first).img);
	}
}
