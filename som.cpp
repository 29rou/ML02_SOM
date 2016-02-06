#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <iostream>
#include <random>
#include <thread>
#define N 2000 //入力画像データの最大数
#define V 900 //特徴ベクトルの次元
#define H 20 //自己組織化マップの大きさ
#define W 20

using namespace std;
using namespace cv;

random_device rnd;
mt19937 mt(rnd());

Mat imgdatas[N] ={};
struct imgdata{float fvex[V]={}; Mat* img={};};
struct imgdata imgd[N] = {};
float somap[H][W][V] = {};
Mat* imgmap[H][W] = {};
inline void toimg(const int &c);
inline void picimg(const int &c,const int &i,const int &j);

void tovec(const std::string &filepath, const int &n){
    Mat src_img = imread(filepath);
    Mat img;
    resize(src_img,img,Size(96,96),CV_INTER_CUBIC);
    cvtColor(img, img, CV_RGB2GRAY);
    HOGDescriptor hog(Size(96,96),Size(32,32),Size(16,16),Size(16,16),9,-1,0.2,true,64);
    std::vector<Point> locations;
    std::vector<float> featureVec;
    hog.compute(img,featureVec,Size(0,0),Size(0,0), locations);
    //std::cout << featureVec.size();
    resize(src_img,src_img,Size(40,40),CV_INTER_CUBIC);
    cvtColor(src_img, src_img, CV_RGB2GRAY,CV_INTER_CUBIC);
    equalizeHist(src_img, src_img);
    normalize(src_img, src_img, 0, 255, NORM_MINMAX);
    for(int i=0;i<n;i++)imgd[n].fvex[i]=featureVec[i];
    imgdatas[n] = src_img;
    imgd[n].img = &imgdatas[n];
}


int fetchdata(const int argc, const char *argv[]){
    namespace fs = boost::filesystem;
    int n=0;
    #ifdef _OPENMP
    #pragma omp parallel for 
    #endif
    for(int i=1; i<argc; i++){
        fs::path path(argv[i]);
        BOOST_FOREACH(const fs::path& p, make_pair(fs::recursive_directory_iterator(path),fs::recursive_directory_iterator())) {
                tovec(p.string(),n++);
        }
    }
    return n;
}

void initialize(const int &c){
    //cout<<c; 
    uniform_int_distribution<> randV(0, c); 
    int temp=0;
    for(int i=0; i<H;i++){
        for(int j=0; j<W;j++){
            temp =randV(mt);
            //cout<<temp<<endl;
            for(int n=0;n<V;n++)somap[i][j][n] =imgd[temp].fvex[n];
            imgmap[i][j] = imgd[temp].img;
        }
    }
}

float findnear(const int &n,int &x,int &y){
    float dist[H][W];
    float min=FLT_MAX;
    vector<int> xs;
    vector<int> ys;
    {
        #ifdef _OPENMP
        #pragma omp parallel for num_threads(4) schedule(static)
        #endif
        for(int i=0; i<H;i++){
            for(int j=0; j<W;j++){
                dist[i][j] = 0;
                for(int k=0; k<V;k+=9){
                    dist[i][j]+=fabsf(imgd[n].fvex[k]-somap[i][j][k]);
                    dist[i][j]+=fabsf(imgd[n].fvex[k+1]-somap[i][j][k+1]);
                    dist[i][j]+=fabsf(imgd[n].fvex[k+2]-somap[i][j][k+2]);
                    dist[i][j]+=fabsf(imgd[n].fvex[k+3]-somap[i][j][k+3]);
                    dist[i][j]+=fabsf(imgd[n].fvex[k+4]-somap[i][j][k+4]);
                    dist[i][j]+=fabsf(imgd[n].fvex[k+5]-somap[i][j][k+5]);
                    dist[i][j]+=fabsf(imgd[n].fvex[k+6]-somap[i][j][k+6]);
                    dist[i][j]+=fabsf(imgd[n].fvex[k+7]-somap[i][j][k+7]);
                    dist[i][j]+=fabsf(imgd[n].fvex[k+8]-somap[i][j][k+8]);
                }
            }
        }//cout<<n;
    }
    //cout<<n;
    for(int i=0; i<H;i++){
        for(int j=0; j<W; j++){
            if(dist[i][j]<=min){
                min = dist[i][j];
                ys.push_back(i);
                xs.push_back(j);
            }
        }
    }
    if(min==FLT_MAX){
        x=0;y=0;
        return -1.0;
    }
    uniform_int_distribution<> randin(0, xs.size()-1); 
    int index = randin(mt);
    //cout<<min<<" "<<xs.size()<<endl;
    y=ys[index];
    x=xs[index];
    return min;
}

void train(const int &n,const int &c,const int x,const int y){
    int dist;
    int vic;
    float tdata[V];
    if(n==N){vic=10;}
    else if(n<(10000)){vic= 8;}
    else if(n<(20000)){vic = 7;}
    else if(n<(40000)){vic = 6;}
    else if(n<(80000)){vic = 5;}
    else if(n<(160000)){vic = 4;}
    else if(n<(320000)){vic = 3;}
    else if(n<(1000000)){vic = 2;}
    else{vic = 1;}
    memcpy(tdata,imgd[n].fvex,sizeof(float)*V);
    #ifdef _OPENMP
    #pragma omp parallel for num_threads(4) schedule(dynamic,1)
    #endif
    for(int i=0;i<H;i++){
        float w;
        for(int j=0;j<W;j++){
            dist = abs(i-y)+abs(j-x);
            if(dist<vic){
                //cout<<dist<<" ";
                if(dist==0&&n>(99*N/100)){for(int k=0; k<V;k++)somap[i][j][k]=tdata[k];}
                else if(dist==0){w = 1;}
                else if(n>(99*N/100)){w = sqrtf((1.0/(float)dist));}
                else if(n>(9*N/10)){w = (1.0/(float)dist);}
                else if(n>(8*N/10)){w = (1.0/((float)dist*2.0));}
                else if(n>(5*N/10)){w = (1.0/((float)dist*3.0));}
                else{w = 1.0/((float)dist*4.0);}
                //cout<<w<<"\n";
                //#ifdef _OPENMP
                //#pragma omp parallel for num_threads(4) schedule(static)
                //#endif
                for(int k=0; k<V;k+=9){
                somap[i][j][k]=somap[i][j][k]+(w*(tdata[k]-somap[i][j][k]));
                somap[i][j][k+1]=somap[i][j][k+1]+(w*(tdata[k+1]-somap[i][j][k+1]));
                somap[i][j][k+2]=somap[i][j][k+2]+(w*(tdata[k+2]-somap[i][j][k+2]));
                somap[i][j][k+3]=somap[i][j][k+3]+(w*(tdata[k+3]-somap[i][j][k+3]));
                somap[i][j][k+4]=somap[i][j][k+4]+(w*(tdata[k+4]-somap[i][j][k+4]));
                somap[i][j][k+5]=somap[i][j][k+5]+(w*(tdata[k+5]-somap[i][j][k+5]));
                somap[i][j][k+6]=somap[i][j][k+6]+(w*(tdata[k+6]-somap[i][j][k+6]));
                somap[i][j][k+7]=somap[i][j][k+7]+(w*(tdata[k+7]-somap[i][j][k+7]));
                somap[i][j][k+8]=somap[i][j][k+8]+(w*(tdata[k+8]-somap[i][j][k+8]));
                }
                picimg(c,i,j);
            }
        }
    }
}

void somcomput(const int c,int &n,const int max){
    uniform_int_distribution<> randc(0, c);
    int x=0; int y=0;
    int temp;
    //#ifdef _OPENMP
    //#pragma omp parallel for num_threads(4) schedule(static)
    //#endif
    for(;n<=max;){
        //cout<<n<<"/"<<max<<endl;
        temp= randc(mt);
        //cout<<temp;
        if(findnear(temp,x,y)>=0)train(temp,c,x,y);
        //#pragma omp atomic
        //n++;
        n++;
    }
}

void showimg(const int c,const int &n,const int max){
    for(;n<=max;){
        //cout<<n<<"/"<<max<<endl;
        toimg(c);
        waitKey(1);
        chrono::milliseconds waittime( 298 );
        this_thread::sleep_for(chrono::milliseconds(waittime));
    }
    toimg(c);
    waitKey(0);
}

inline void som(const int &c,const int &max){
    int n=0;
    //cout<<flush<<c<<" "<<n<<"/"<<max<<endl;
    thread t1(somcomput,c,ref(n),max);
    thread t2(showimg,c,ref(n),max);
    t1.join();
    t2.join();
}

inline void picimg(const int &c,const int &i,const int &j){
    uniform_int_distribution<> randk(0, c); 
    int count = c*10;
    int x=0;
    float min=FLT_MAX;
    vector<int> index;
    //int tmp=randk(mt);
    for(int k=0;k<count;k++){
        float dist = 0;
        int tmp=randk(mt);
        for(int l=0; l<V;l+=1){
            dist+=fabsf(imgd[tmp].fvex[l]-somap[i][j][l]);
        }
        if(dist<=min){
        min = dist;
        index.push_back(tmp);
        }
    }
    uniform_int_distribution<> randi(0, index.size()-1); 
    //int tmp = index[randi(mt)];
    //cout<<tmp<<endl;;
    imgmap[i][j] = imgd[index[randi(mt)]].img;
    index.clear();
}

inline void toimg(const int &c){
    Mat combined_img(Size(W*imgmap[0][0]->cols,H*imgmap[0][0]->rows),CV_8U);
    int width = imgmap[0][0]->cols;
    int height = imgmap[0][0]->rows;
    Rect roi_rect;
    roi_rect.width = width;
    roi_rect.height = height;
    for(int i=0;i<H;i++){
        for(int j=0;j<W;j++){
             Mat roi(combined_img, roi_rect);
             imgmap[i][j]->copyTo(roi);
             roi_rect.x +=width;
        }
        roi_rect.x =0;
        roi_rect.y +=height;
    }
    imshow("SOM",combined_img);
}


int main(const int argc, const char *argv[]){
    const int count = fetchdata(argc,argv);
    initialize(count-1);
    //namedWindow("SOM");
    //constexpr int max = 10000;
    som(count-1,10000*10000);
    //imshow("test",*imgmap[0][0]);
    //out <<imgmap[1][0]<<" "<<imgmap[10][0];
}
