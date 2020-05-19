-#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QFile>
#include <QFileDialog>
#include <opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <asmopencv.h>
#include <svm.cpp>
#include <svm.h>
#define Malloc(type,n) (type*)malloc((n)*sizeof(type))
struct svm_parameter myParams;
struct svm_problem myProblem;
struct svm_model* myModel;
struct svm_node* x_space;
using namespace std;
using namespace cv;
MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{

    ui->setupUi(this);
    //this->setFixedSize(1280,720);
    // testing libSVM
      myParams.svm_type = C_SVC;
      myParams.kernel_type = RBF;
      myParams.degree =3;
      myParams.gamma =0.5;
      myParams.cache_size =100;
      myParams.C =1;
      myParams.eps=1e-3;
      myParams.p=0.1;
      myParams.shrinking =1;
      myParams.probability =0;
      myParams.nr_weight = 0;
      myParams.weight_label =NULL;
      myParams.weight_label =NULL;
      myProblem.l= 4;
      double matrix[4][2];
      matrix[0][0]=1;
      matrix[0][1]=1;

      matrix[1][0]=1;
      matrix[1][1]=0;

      matrix[2][0]=0;
      matrix[2][1]=1;

      matrix[3][0]=0;
      matrix[3][1]=0;


      svm_node** x= Malloc(svm_node*,myProblem.l);
      for (int row=0;row<myProblem.l;row++)
      { svm_node* x_space= Malloc(svm_node,3);
          for (int col =0;col <2; col++)
          { x_space[col].index=col;
              x_space[col].value=matrix[row][col];
      }
      x_space[2].index = -1;
      x[row]= x_space;
    }

      myProblem.x = x;
      myProblem.y=Malloc(double,myProblem.l);
      myProblem.y[0]=-1;
      myProblem.y[1]= 1;
      myProblem.y[2]= 1;
      myProblem.y[3]=-1;

      svm_model* model = svm_train(&myProblem,&myParams);

      svm_node* testnode = Malloc(svm_node,3);
      testnode[0].index =0;
      testnode[0].value =0;
      testnode[1].index =1;
      testnode[1].value =0;
      testnode[2].index =-1;
      double retval = svm_predict(model,testnode);
      printf("retval: %f\n",retval);

}

MainWindow::~MainWindow()
{

    delete ui;
}

void MainWindow::on_pushButton_clicked()
{
loadImage();
}

void MainWindow::loadImage()
{
    QString const inputPath = QFileDialog::getOpenFileName(this, "Open a file", "directoryToOpen",
            "Images (*.png *.xpm *.jpg);;Text files (*.txt);;XML files (*.xml)");
   //std::string path = inputPath.toLocal8Bit().constData();

imread(inputPath.toStdString());
   Mat img = imread(inputPath.toStdString(),CV_LOAD_IMAGE_GRAYSCALE);
       if(!img.data )
       {
        ui->label_8->setText("Could not open/find the image");

       }



       Mat img_bw; //array to store binarised image
       cv::threshold(img, img_bw, 0, 255, CV_THRESH_BINARY_INV | CV_THRESH_OTSU);




   QImage qImage = ASM::cvMatToQImage(img_bw);
   QPixmap pixmap = QPixmap::fromImage(qImage);
  //ui->label_8->setPixmap(pixmap.scaled(QSize(1280,720)));
  ui->label_8->setPixmap(pixmap.scaled(QSize(400,400),Qt::KeepAspectRatio));

}
void MainWindow::chooseDirectory()
{
    QString dir = QFileDialog::getExistingDirectory(this, tr("Open Directory"),
                                                    "/home",
                                                    QFileDialog::ShowDirsOnly
                                                    | QFileDialog::DontResolveSymlinks);




}

void MainWindow::on_pushButton_2_clicked()
{
    chooseDirectory();
}
