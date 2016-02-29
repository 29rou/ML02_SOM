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

void somap::setw(const int &count, const std::pair<int,int> &ptr, const int &x,const int &y) {
	int dist = abs(y - ptr.second) + abs(x - ptr.first);
	if (count < 5000000) {
		if (count <= 100 && dist < H*1.5)this->weight = 0.5;
		else if (count <= 100000 && dist < H)this->weight = 0.7;
		else if (count <= 500000 && dist < H / 1.5)this->weight = 0.9;
		else if (count <= 1000000 && dist < H / 2)this->weight = 0.7;
		else if (count <= 1500000 && dist < H / 3)this->weight = 0.5;
		else if (count <= 2000000 && dist < H / 4)this->weight = 0.3;
		else if (count <= 3200000 && dist < H / 5)this->weight = 0.1;
		else if (count <= 4000000 && dist < H / 6)this->weight = 0.01;
		else if (dist < 4)this->weight = 1.00;
		else this->weight = -1;
	}
	else if (count < 10000000) {
		switch (dist) {
		case 0:
		case 1:
		case 2:
		case 3:
			this->weight = 0.01;
			break;
		case 4:
		case 5:
			this->weight = 0.009;
			break;
		case 6:
		case 7:
			this->weight = 0.008;
			break;
		case 8:
		case 9:
			this->weight = 0.007;
			break;
		case 10:
		case 11:
			this->weight = 0.006;
			break;
		case 12:
		case 13:
		case 14:
		case 15:
			this->weight = 0.005;
			break;
		default:
			this->weight = -1;
			break;
		}
	}
	else {
		switch (dist) {
		case 0:
			this->weight = 0.001;
			break;
		case 1:
		case 2:
			this->weight = 0.0005;
			break;
		case 3:
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

void somap::getnearlist(imgdatas & imgdata,std::vector<Mat*> &matlist)
{
	std::vector<std::pair<int, float>> tmp;
	tmp.reserve(imgdata.size());
	auto forsort = [](std::pair<int, float> &x, std::pair<int, float> &y) -> bool {
		if (x.second > y.second)return false;
		if (x.second < y.second)return true;
		return false;
	};
	auto forunique = [](std::pair<int, float> &x, std::pair<int, float> &y) -> bool {
		if (x.second == y.second)return true;
		return false;
	};
	for (int i = 0; i < imgdata.size(); i++)tmp.push_back({ i,this->getDistance(imgdata.at(i)) });
	std::sort(tmp.begin(), tmp.end(), forsort);
	auto result = unique(tmp.begin(), tmp.end(), forunique);
	tmp.erase(result, tmp.end());
	matlist.reserve(imgdata.size());
	for (auto &i : tmp) {
		matlist.push_back(&imgdata.at(i.first).img);
	}
}

void somap::getnearlist(imgdatas & imgd, std::vector<imgdata*> &matlist)
{
	std::vector<std::pair<int, float>> tmp;
	tmp.reserve(imgd.size());
	auto forsort = [](std::pair<int, float> &x, std::pair<int, float> &y) -> bool {
		if (x.second > y.second)return false;
		if (x.second < y.second)return true;
		return false;
	};
	auto forunique = [](std::pair<int, float> &x, std::pair<int, float> &y) -> bool {
		if (x.second == y.second)return true;
		return false;
	};
	for (int i = 0; i < imgd.size(); i++)tmp.push_back({ i,this->getDistance(imgd.at(i)) });
	std::sort(tmp.begin(), tmp.end(), forsort);
	matlist.reserve(imgd.size());
	for (auto &i : tmp) {
		matlist.push_back(&imgd.at(i.first));
	}
}
