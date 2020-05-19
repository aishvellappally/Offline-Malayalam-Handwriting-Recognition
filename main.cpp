//
//  main.cpp
//  major_project3
//
//  Created by Aiswarya on 11/05/16.
//  Copyright © 2016 Aiswarya. All rights reserved.
//

//Segmentation
#include <opencv2/highgui/highgui.hpp> //header for threshold fn
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <sstream>
#include<fstream>
#include "svm.h"
#define Malloc(type,n) (type*)malloc((n)*sizeof(type))
struct svm_parameter myParams;
struct svm_problem myProblem;
struct svm_model* myModel;
struct svm_node* x_space;
int k1;
using namespace cv;
using namespace std;
ofstream myfile;
void segment(Mat img) {

    
    Mat img_bw; //array to store binarised image
    
    threshold(img, img_bw, 0, 255, CV_THRESH_BINARY_INV | CV_THRESH_OTSU); //for otsu thresholding and inverting
    
    int r,c;
    r=img_bw.rows;
    
    c=img_bw.cols;
    int S[r];
    for(int i=0;i<r;i++)
    {
        S[i]=0;
    }
    
    for(int i=0;i<r;i++)
    {
        for(int j=0;j<c;j++)
        {
            S[i]+=img_bw.at<uchar>(i,j);
        }
    }
    
    int j=0;
    for(int i=0;i<r;i++)
    {
        if(S[i]==0)
        {
            if(i!=r-1)
            {
                if(S[i+1]!=0)
                {
                    j=j+1;
                }
            }
        }
    }
    int r1=r/j;
    int k=0;
    int p=0;
    Mat g(r1,c,CV_8UC1,Scalar(0));
    for(int i=1;i<r;i++)
    {
        if(S[i]!=0) //if current line is not empty
        {
            if (S[i-1]==0) //if previous line is empty - start of new line
            {
                for(int k=p;k<r1;k++)
                {
                    for(int l=0;l<c;l++)
                    {
                        g.at<uchar>(k,l)=0;
                    }
                }
                k=k+1;
                p=0; //reset row number value
                ostringstream fname;
                fname<<"/Users/aiswarya/Desktop/test/segline"<<k<<".png";
                
                imwrite(fname.str(),g);
                
            }
            
            if(p<r1)
            {
                for(int l=0;l<c;l++)
                {
                    g.at<uchar>(p,l)=img_bw.at<uchar>(i,l);
                }
                p=p+1;
            }
            else
                continue;
        }
        
    }
    for(int k=p;k<r1;k++)
    {
        for(int l=0;l<c;l++)
        {
            g.at<uchar>(k,l)=0;
        }
    }
    ostringstream fname;
    fname<<"/Users/aiswarya/Desktop/test/segline"<<k+1<<".png";
    imwrite(fname.str(),g);
    
    int k1=0;
    int l=0;
    int sum[r1];
    for (int i=0;i<k+1;i++)
    {
        ostringstream fname;
        fname<<"/Users/aiswarya/Desktop/test/segline"<<i+1<<".png";
        Mat g1=imread(fname.str(),CV_LOAD_IMAGE_UNCHANGED);
        Mat g2;
        threshold(g1,g2, 0, 255, CV_THRESH_OTSU);
        int row,col;
        row=g2.rows;
        col=g2.cols;
        for (int i=0;i<row;i++)
        {
            sum[i]=0;
        }
        for(int i=0;i<row;i++)
        {
            for(int j=0;j<col;j++)
            {
                sum[i]+=g2.at<uchar>(i,j);
                
            }
        }
        
        for(int k=0;k<row;k++)
        {
            if(sum[k]!=0 && sum[k+1]==0)
            {
                l=k+1;
            }
            
        }
        if(l>=(row/7))
        {
            int S1[col];
            for(int i=0;i<col;i++)
            {
                S1[i]=0;
            }
            for(int i=0;i<col;i++)
            {
                for(int j1=0;j1<row;j1++)
                {
                    S1[i]+=g2.at<uchar>(j1,i);
                }
                
            }
            
            int p=0;
            int q=0;
            for(int j1=2;j1<col;j1++)
            {
                if(S1[j1]!=0 && S1[j1-1]==0)
                {
                    p=j1-2;
                }
                if(S1[j1]==0 && S1[j1-1]!=0)
                {
                    q=j1;
                    k1=k1+1;
                    k=0;
                    Mat f2(l,q-p+1,CV_8UC1,Scalar(0));
                    for (int t=p-1;t<q-1;t++)
                    {
                        k=k+1;
                        for(int i=0;i<l;i++)
                        {
                            f2.at<uchar>(i,k)=g2.at<uchar>(i,t);
                        }
                        
                    }
                    Mat f3(20,20,CV_8UC1,Scalar(0));
                    resize(f2,f2,f3.size());
                    ostringstream name;
                    if(f2.rows!=0&&f2.cols!=0)
                        
                    {
                        ostringstream name;
                        name<<"/Users/aiswarya/Desktop/test/schar"<<k1<<".png";
                        imwrite(name.str(),f2);
                    }
                    
                }
            }
            /*if(row!=0&&col!=0)
             {
             ostringstream name1;
             name1<<"/Users/aiswarya/Desktop/new/doc"<<doc_number<<"sline"<<"_"<<i+1<<".png";
             imwrite(name1.str(),g2);
             }*/
        }
    }
    
    
}



double testing(Mat f,struct svm_model *model)
{
    int r,c;
    r=f.rows;
    c=f.cols;
    int t=0;
    svm_node* testnode = Malloc(svm_node,401);
    
    for(int i=0;i<r;i++)
    {
        for(int j=0;j<c;j++)
        {
            testnode[t].index=t;
            testnode[t].value=((f.at<uchar>(i,j))/255);
            t++;
        }
            
    }
    testnode[400].index=-1;
    
    double retval = svm_predict(model,testnode);
    return retval;
    

}
void translate(int retval) {
    if(retval == 1) {
    } else if(retval == 1) {
        myfile.open("/User/Desktop/output.txt",ios::app);
        cout<< "അ ";
        myfile.close();
    }
    else if(retval == 2) {
        myfile.open("/User/Desktop/output.txt",ios::app);
        cout<< "ആ ";
        myfile.close();
    }else if(retval == 3) {
        myfile.open("/User/Desktop/output.txt",ios::app);
        cout<< "ഇ ";
        myfile.close();
    }else if(retval == 4) {
        myfile.open("/User/Desktop/output.txt",ios::app);
        cout<< "ഉ ";
        myfile.close();
    }else if(retval == 5) {
        myfile.open("/User/Desktop/output.txt",ios::app);
        cout<< "ഋ ";
        myfile.close();
    }else if(retval == 6) {
        myfile.open("/User/Desktop/output.txt",ios::app);
        cout<< "എ ";
        myfile.close();
    }else if(retval == 7) {
        myfile.open("/User/Desktop/output.txt",ios::app);
        cout<< "ഏ ";
        myfile.close();
    }else if(retval == 8) {
        myfile.open("/User/Desktop/output.txt",ios::app);
        cout<< "ഒ ";
        myfile.close();
    }else if(retval == 9) {
        myfile.open("/User/Desktop/output.txt",ios::app);
        cout<< "ക ";
        myfile.close();
    }else if(retval == 10) {
        myfile.open("/User/Desktop/output.txt",ios::app);
        cout<< "ഖ ";
        myfile.close();
    }else if(retval == 11) {
        myfile.open("/User/Desktop/output.txt",ios::app);
        cout<< "ഗ ";
        myfile.close();
    }else if(retval == 12) {
        myfile.open("/User/Desktop/output.txt",ios::app);
        cout<< "ഘ ";
        myfile.close();
    }else if(retval == 13) {
        myfile.open("/User/Desktop/output.txt",ios::app);
        cout<< "ങ ";
        myfile.close();
    }else if(retval == 14) {
        myfile.open("/User/Desktop/output.txt",ios::app);
        cout<< "ച ";
        myfile.close();
    }else if(retval == 15) {
        myfile.open("/User/Desktop/output.txt",ios::app);
        cout<< "ഛ ";
        myfile.close();
    }else if(retval == 16) {
        myfile.open("/User/Desktop/output.txt",ios::app);
        cout<< "ജ ";
        myfile.close();
    }else if(retval == 17) {
        myfile.open("/User/Desktop/output.txt",ios::app);
        cout<< "ഝ ";
        myfile.close();
    }else if(retval == 18) {
        myfile.open("/User/Desktop/output.txt",ios::app);
        cout<< "ഞ ";
        myfile.close();
    }else if(retval == 19) {
        myfile.open("/User/Desktop/output.txt",ios::app);
        cout<< "ട ";
        myfile.close();
    }else if(retval == 20) {
        myfile.open("/User/Desktop/output.txt",ios::app);
        cout<< "ഠ ";
        myfile.close();
    }else if(retval == 21) {
        myfile.open("/User/Desktop/output.txt",ios::app);
        cout<< "ഡ ";
        myfile.close();
    }else if(retval == 22) {
        myfile.open("/User/Desktop/output.txt",ios::app);
        cout<< "ഢ ";
        myfile.close();
    }else if(retval == 23) {
        myfile.open("/User/Desktop/output.txt",ios::app);
        cout<< "ണ ";
        myfile.close();
    }else if(retval == 24) {
        myfile.open("/User/Desktop/output.txt",ios::app);
        cout<< "ത ";
        myfile.close();
    }else if(retval == 25) {
        myfile.open("/User/Desktop/output.txt",ios::app);
        cout<< "ഥ ";
        myfile.close();
    }else if(retval == 26) {
        myfile.open("/User/Desktop/output.txt",ios::app);
        cout<< "ദ ";
        myfile.close();
    }else if(retval == 27) {
        myfile.open("/User/Desktop/output.txt",ios::app);
        cout<< "ധ ";
        myfile.close();
    }else if(retval == 28) {
        myfile.open("/User/Desktop/output.txt",ios::app);
        cout<< "ന ";
        myfile.close();
    }else if(retval == 29) {
        myfile.open("/User/Desktop/output.txt",ios::app);
        cout<< "പ ";
        myfile.close();
    }else if(retval == 30) {
        myfile.open("/User/Desktop/output.txt",ios::app);
        cout<< "ഫ ";
        myfile.close();
    }else if(retval == 31) {
        myfile.open("/User/Desktop/output.txt",ios::app);
        cout<< "ബ ";
        myfile.close();
    }else if(retval == 32) {
        myfile.open("/User/Desktop/output.txt",ios::app);
        cout<< "ഭ ";
        myfile.close();
    }else if(retval == 33) {
        myfile.open("/User/Desktop/output.txt",ios::app);
        cout<< "മ ";
        myfile.close();
    }else if(retval == 34) {
        myfile.open("/User/Desktop/output.txt",ios::app);
        cout<< "യ ";
        myfile.close();
    }else if(retval == 35) {
        myfile.open("/User/Desktop/output.txt",ios::app);
        cout<< "ര ";
        myfile.close();
    }else if(retval == 36) {
        myfile.open("/User/Desktop/output.txt",ios::app);
        cout<< "ല ";
        myfile.close();
    }else if(retval == 37) {
        myfile.open("/User/Desktop/output.txt",ios::app);
        cout<< "വ ";
        myfile.close();
    }else if(retval == 38) {
        myfile.open("/User/Desktop/output.txt",ios::app);
        cout<< "ശ ";
        myfile.close();
    }else if(retval == 39) {
        myfile.open("/User/Desktop/output.txt",ios::app);
        cout<< "ഷ ";
        myfile.close();
    }else if(retval == 40) {
        myfile.open("/User/Desktop/output.txt",ios::app);
        cout<< "സ ";
        myfile.close();
    }else if(retval == 41) {
        myfile.open("/User/Desktop/output.txt",ios::app);
        cout<< "ഹ";
        myfile.close();
    }else if(retval == 42) {
        myfile.open("/User/Desktop/output.txt",ios::app);
        cout<< "ള ";
        myfile.close();
    }else if(retval == 43) {
        myfile.open("/User/Desktop/output.txt",ios::app);
        cout<< "ഴ ";
        myfile.close();
    }else if(retval == 44) {
        myfile.open("/User/Desktop/output.txt",ios::app);
        cout<< "റ ";
        myfile.close();
    }else if(retval == 45) {
        myfile.open("/User/Desktop/output.txt",ios::app);
        cout<< "ർ ";
        myfile.close();
    }else if(retval == 46) {
        myfile.open("/User/Desktop/output.txt",ios::app);
        cout<< "ൻ ";
        myfile.close();
    }else if(retval == 47) {
        myfile.open("/User/Desktop/output.txt",ios::app);
        cout<< "ൾ ";
        myfile.close();
    }else if(retval == 48) {
        myfile.open("/User/Desktop/output.txt",ios::app);
        cout<< "ൺ ";
        myfile.close();
    }else if(retval == 49) {
        myfile.open("/User/Desktop/output.txt",ios::app);
        cout<< "ൽ ";
        myfile.close();
    }else if(retval == 50) {
        myfile.open("/User/Desktop/output.txt",ios::app);
        cout<< "ാ ";
        myfile.close();
    }else if(retval == 51) {
        myfile.open("/User/Desktop/output.txt",ios::app);
        cout<< "ി ";
        myfile.close();
    }else if(retval == 52) {
        myfile.open("/User/Desktop/output.txt",ios::app);
        cout<< "ീ ";
        myfile.close();
    }else if(retval == 53) {
        myfile.open("/User/Desktop/output.txt",ios::app);
        cout<< "ു ";
        myfile.close();
    }else if(retval == 54) {
        myfile.open("/User/Desktop/output.txt",ios::app);
        cout<< "ൂ ";
        myfile.close();
    }else if(retval == 55) {
        myfile.open("/User/Desktop/output.txt",ios::app);
        cout<< "ൃ ";
        myfile.close();
    }else if(retval == 56) {
        myfile.open("/User/Desktop/output.txt",ios::app);
        cout<< "ൗ ";
        myfile.close();
    }else if(retval == 57) {
        myfile.open("/User/Desktop/output.txt",ios::app);
        cout<< "െ ";
        myfile.close();
    }else if(retval == 58) {
        myfile.open("/User/Desktop/output.txt",ios::app);
        cout<< "േ ";
        myfile.close();
    }else if(retval == 59) {
        myfile.open("/User/Desktop/output.txt",ios::app);
        cout<< "ൈ ";
        myfile.close();
        
    }else if(retval == 60) {
        myfile.open("/User/Desktop/output.txt",ios::app);
        cout<< "ം ";
        myfile.close();
        
    }else if(retval == 61) {
        myfile.open("/User/Desktop/output.txt",ios::app);
        cout<< "ഃ ";
        myfile.close();
        
    }else if(retval == 62) {
        myfile.open("/User/Desktop/output.txt",ios::app);
        cout<< "്";
        myfile.close();
    }
}



int main(int argc, const char * argv[]) {
    
    ofstream myfile("/User/Desktop/output.txt");

    myfile << "Writing this into a file!";
    myfile.close();
    

    
        Mat img= imread("/Users/aiswarya/Documents/old desktop/major project/matlab_svm/test2.jpg",CV_LOAD_IMAGE_GRAYSCALE); //'0' also reads image in BGR format
        
        if(! img.data )
        {
            cout <<  "Could not open or find the image" << endl ;
            return -1;
        }
        
        segment(img) ;
        cout<<"abcd";
    
    struct svm_model *loaded_model;
    loaded_model=svm_load_model("/Users/aiswarya/Desktop/project/model.txt");
    double retval;
    for(int i=1;i<=11;i++)
    {
        ostringstream name;
        name<<"/Users/aiswarya/Documents/old desktop/test/schar"<<i<<".png";

        Mat testimage=imread(name.str(),CV_LOAD_IMAGE_GRAYSCALE);
        if(! testimage.data )
        {
            cout <<  "Could not open or find the image" << endl ;
            return -1;
        }

        Mat test1;
        threshold(testimage, test1, 0, 255, CV_THRESH_BINARY_INV | CV_THRESH_OTSU);
        retval=testing(test1,loaded_model);
        translate(retval);
        
    }
    
    return 0;
    
    
    
}



