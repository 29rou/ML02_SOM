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
#define N 6000 //入力画像データの最大数
#define V 900 //特徴ベクトルの次元
#define H 17 //自己組織化マップの大きさ
#define W 17

using namespace std;
using namespace cv;

random_device rnd;
mt19937 mt(rnd());

Mat imgdatas[N]={};
struct imgdata{float fvex[V]={}; Mat* img={};};
struct imgdata imgd[N] = {};
float somap[H][W][V] = {};
Mat* imgmap[H][W] = {};

extern void toimg(const int &count);

void tovec(const std::string &filepath, const int &n){
    Mat src_img = imread(filepath);
    Mat img;
    resize(src_img,img,Size(96,96));
    cvtColor(img, img, CV_RGB2GRAY);
    HOGDescriptor hog(Size(96,96),Size(32,32),Size(16,16),Size(16,16),9,-1,0.2,true,64);
    std::vector<Point> locations;
    std::vector<float> featureVec;
    hog.compute(img,featureVec,Size(0,0),Size(0,0), locations);
    //std::cout << featureVec.size();
    resize(src_img,src_img,Size(50,50));
    cvtColor(src_img, src_img, CV_RGB2GRAY);
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
        BOOST_FOREACH(const fs::path& p, std::make_pair(fs::recursive_directory_iterator(path),fs::recursive_directory_iterator())) {
                tovec(p.string(),n++);
        }
    }
    return n;
}

void initialize(const int &c){
    cout<<c; 
    uniform_int_distribution<> randV(0, c); 
    int temp=0;
    for(int i=0; i<H;i++){
        for(int j=0; j<W;j++){
            temp =randV(mt);
            cout<<temp<<endl;
            for(int n=0;n<V;n++)somap[i][j][n] =imgd[temp].fvex[n];
            //imgmap[i][j] = imgd[temp].img;
        }
    }
}

float findnear(const int &n,int &x,int &y){
    double dist[H][W];
    double min=DBL_MAX;
    vector<int> xs;
    vector<int> ys;
    {
        #ifdef _OPENMP
        #pragma omp parallel for 
        #endif
        for(int i=0; i<H;i++){
            for(int j=0; j<W;j++){
                dist[i][j] = 0;
                for(int k=0; k<V;k+=9){
                    dist[i][j]+=powl((double)(imgd[n].fvex[k]-somap[i][j][k]),2);
                    dist[i][j]+=powl((double)(imgd[n].fvex[k+1]-somap[i][j][k+1]),2);
                    dist[i][j]+=powl((double)(imgd[n].fvex[k+2]-somap[i][j][k+2]),2);
                    dist[i][j]+=powl((double)(imgd[n].fvex[k+3]-somap[i][j][k+3]),2);
                    dist[i][j]+=powl((double)(imgd[n].fvex[k+4]-somap[i][j][k+4]),2);
                    dist[i][j]+=powl((double)(imgd[n].fvex[k+5]-somap[i][j][k+5]),2);
                    dist[i][j]+=powl((double)(imgd[n].fvex[k+6]-somap[i][j][k+6]),2);
                    dist[i][j]+=powl((double)(imgd[n].fvex[k+7]-somap[i][j][k+7]),2);
                    dist[i][j]+=powl((double)(imgd[n].fvex[k+8]-somap[i][j][k+8]),2);
                }
                dist[i][j] = sqrtl(dist[i][j]);
            }
        }
    }
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
        x=INT_MAX;y=INT_MAX;
        return (float)min;
    }
    uniform_int_distribution<> randin(0, xs.size()-1); 
    int index = randin(mt);
    //cout<<min<<" "<<xs.size()<<endl;
    y=ys[index];
    x=xs[index];
    return min;
}

void train(const int &n,const int x,const int y){
    double dist;
    double vic;
    float tdata[V];
    if(n>(9*N/10)){vic=sqrtl(2.0*powl(20.0,2));}
    else if(n>(8*N/10)){vic = sqrtl(2.0*powl(15.0,2));}
    else if(n>(7*N/10)){vic = sqrtl(2.0*powl(10.0,2));}
    else if(n>(5*N/10)){vic = sqrtl(2.0*powl(5.0,2));}
    else if(n>(N/10)){vic = sqrtl(2.0*powl(2.0,2));}
    else{vic = sqrtl(2.0*powl(1.0,2));}
    memcpy(tdata,imgd[n].fvex,sizeof(float)*V);
    #ifdef _OPENMP
    #pragma omp parallel for 
    #endif
    for(int i=0;i<H;i++){
        double w;
        for(int j=0;j<W;j++){
            dist = sqrtl((powl(i-y,2)+powl(j-x,2)));
            if(dist<vic){
                //cout<<dist<<" ";
                if(dist==0&&n>(9*N/10)){for(int k=0; k<V;k++)somap[i][j][k]=tdata[k];}
                else if(dist==0){w = 1;}
                else if(n>(7*N/10)){w = sqrtl((1.0/(double)dist));}
                else if(n>(3*N/10)){w = (1.0/(double)dist);}
                else{w = 1.0/(dist*2.0);}
                //cout<<w<<"\n";
                for(int k=0; k<V;k+=9){
                somap[i][j][k]=(double)somap[i][j][k]+(w*(double)(tdata[k]-somap[i][j][k]));
                somap[i][j][k+1]=(double)somap[i][j][k+1]+(w*(double)(tdata[k+1]-somap[i][j][k+1]));
                somap[i][j][k+2]=(double)somap[i][j][k+2]+(w*(double)(tdata[k+2]-somap[i][j][k+2]));
                somap[i][j][k+3]=(double)somap[i][j][k+3]+(w*(double)(tdata[k+3]-somap[i][j][k+3]));
                somap[i][j][k+4]=(double)somap[i][j][k+4]+(w*(double)(tdata[k+4]-somap[i][j][k+4]));
                somap[i][j][k+5]=(double)somap[i][j][k+5]+(w*(double)(tdata[k+5]-somap[i][j][k+5]));
                somap[i][j][k+6]=(double)somap[i][j][k+6]+(w*(double)(tdata[k+6]-somap[i][j][k+6]));
                somap[i][j][k+7]=(double)somap[i][j][k+7]+(w*(double)(tdata[k+7]-somap[i][j][k+7]));
                somap[i][j][k+8]=(double)somap[i][j][k+8]+(w*(double)(tdata[k+8]-somap[i][j][k+8]));
                }
            }
        }
    }
}

void som(const int &c,const int &max){
    uniform_int_distribution<> randc(0, c);
    int x,y;
    int temp;
    for(int n=0;n<max;n+=1){
        x=0; y=0;
        temp= randc(mt);
        findnear(temp,x,y);
        //cout<<n<<" "<<" "<<x<<" "<<y<<endl;
        if(x==INT_MAX&&y==INT_MAX)continue;
        train(temp,x,y);
    }
}

void picimg(const int &c){
    static float dist[H][W][N];
    for(int i=0; i<H;i++)for(int j=0; j<W;j+=1)for(int k=0;k<c;k++)dist[i][j][k]=0;
    float min;
    vector<int> index;
    #ifdef _OPENMP
    #pragma omp parallel for 
    #endif
    for(int i=0; i<H;i++){
        for(int j=0; j<W;j+=1){
            for(int k=0;k<c;k++){             
                for(int l=0; l<V;l+=1){
                    dist[i][j][k]+=fabsf(imgd[k].fvex[l]-somap[i][j][l]);
                }
            }
        }
    }
    for(int i=0; i<H;i++){
        for(int j=0; j<W; j++){
            min=FLT_MAX;
            for(int k=0;k<c;k++){
                if(dist[i][j][k]<=min){
                    //cout<<dist[i][j][k]<<endl;
                    min = dist[i][j][k];
                    index.push_back(k);
                }
            }
            uniform_int_distribution<> randi(0, index.size()-1); 
            //cout<<imgd[index[randi(mt)]].img<<endl;
            imgmap[i][j] = imgd[index[randi(mt)]].img;
            index.clear();
        }
    }
}

void toimg(const int &count){
    Mat combined_img,tmp[H];
    picimg(count-1);
    #ifdef _OPENMP
    #pragma omp parallel for 
    #endif
    for(int i=0;i<H;i++){
        hconcat(*imgmap[i],W,tmp[i]);
    }
    vconcat(tmp,H,combined_img);
    imshow("SOM",combined_img);
}


int main(const int argc, const char *argv[]){
    //cout<<(sizeof(imgd))<<"\n";
    const int count = fetchdata(argc,argv);
    initialize(count-1);
    som(count-1,10000*1000);
    namedWindow("SOM");
    toimg(count);
    //imshow("test",*imgmap[0][0]);
    //out <<imgmap[1][0]<<" "<<imgmap[10][0];
    waitKey(0);
}
