#include <iostream>
#include "hcblaslib.h"
#include <cstdlib> 
#include "hc_short_vector.hpp"
#include <cblas.h>
#include <unistd.h>
#include "hc_am.hpp"

using namespace std;
int main(int argc, char** argv)
{
    /*  HCBLAS Implementation */
    hc::accelerator accl;
    Hcblaslibrary hc(&accl);  
    if (argc < 5){
        cout<<"No sufficient commandline arguments specified"<<"argc :"<<argc<<endl;
        return -1;
    }
    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int isTransA = (atoi(argv[3]));
    int Imple_type = (atoi(argv[4]));
    int row, col;
    bool ispassed = 1;
    float alpha = 1;
    float beta = 1;
    long lda;
    int incX = 1;
    int incY = 1;
    long xOffset = 0;
    long yOffset = 0;
    long aOffset = 0;
    int batchSize = 128;
    long lenx,  leny;
    hcblasStatus status;
    hcblasTranspose typeA;
    hcblasOrder hcOrder = ColMajor;
    if(isTransA == 0){
        row = N;
        col = M;
        lda = M;
        typeA = NoTrans;
    }
    else{
        row = M;
        col = N;
        lda = N;
        typeA = Trans;
    }
    /* CBLAS Implementation */
    CBLAS_ORDER order;
    CBLAS_TRANSPOSE transa;
    order = CblasColMajor;
    transa = (typeA == NoTrans)? CblasNoTrans : CblasTrans;
    lenx = 1 + (row - 1) * abs(incX);
    leny = 1 + (col - 1) * abs(incY);
    long X_batchOffset = row;
    long Y_batchOffset = col;
    long A_batchOffset = row * col;
    std::vector<hc::accelerator>acc = hc::accelerator::get_all();
    accelerator_view accl_view = (acc[1].get_default_view());


/* Implementation type I - Inputs and Outputs are HCC float array containers */

    if(Imple_type == 1) {
        float *x = (float*)calloc( lenx , sizeof(float));
        float *y = (float*)calloc( leny , sizeof(float));
        float *A = (float *)calloc( lenx * leny , sizeof(float));
        float *ycblas = (float *)calloc( leny , sizeof(float));
        float* devA = hc::am_alloc(sizeof(float) * lenx * leny, acc[1], 0);
        float* devX = hc::am_alloc(sizeof(float) * lenx, acc[1], 0);
        float* devY = hc::am_alloc(sizeof(float) * leny, acc[1], 0);
        for(int i = 0;i < lenx;i++) {
           x[i] = rand() % 10;
        }
        for(int i = 0;i< lenx * leny;i++) {
           A[i] = rand() % 25;
        }
#ifdef PROFILE
        for(int iter=0; iter<10; iter++) {
#endif
        for(int i = 0;i < leny;i++) {
            y[i] = rand() % 15;
            ycblas[i] = y[i];
        }
        accl_view.copy(A, devA, lenx * leny * sizeof(float));
        accl_view.copy(x, devX, lenx * sizeof(float));
        accl_view.copy(y, devY, leny * sizeof(float));
        status =  hc.hcblas_sgemv(accl_view, hcOrder, typeA, M, N, alpha, devA, aOffset, lda, devX, xOffset, incX, beta, devY, yOffset, incY);
        accl_view.copy(devY, y, leny * sizeof(float));
        lda = (hcOrder)? M: N;
        cblas_sgemv( order, transa, M, N, alpha, A, lda , x, incX, beta, ycblas, incY );
        for(int i =0; i < leny; i ++){
            if (y[i] != ycblas[i]){
                ispassed = 0;
                cout <<" HCSGEMV[" << i<< "] " << y[i] << " does not match with CBLASSGEMV[" << i <<"] "<< ycblas[i] << endl;
                break;
            }
            else
                continue;
        }
        if(!ispassed) cout << "TEST FAILED" << endl; 
        if(status) cout << "TEST FAILED" << endl; 
#ifdef PROFILE
        }
#endif
        free(x);
        free(y);
        free(A);
        free(ycblas);
        hc::am_free(devA);
        hc::am_free(devX);
        hc::am_free(devY);
    }

/* Implementation type II - Inputs and Outputs are HCC float array containers with batch processing */

    else{
        float *xbatch = (float*)calloc( lenx * batchSize, sizeof(float));
        float *ybatch = (float*)calloc( leny * batchSize, sizeof(float));
        float *Abatch = (float *)calloc( lenx * leny * batchSize, sizeof(float));
        float *ycblasbatch = (float *)calloc( leny * batchSize, sizeof(float));
        float* devXbatch = hc::am_alloc(sizeof(float) * lenx * batchSize, acc[1], 0);
        float* devYbatch = hc::am_alloc(sizeof(float) * leny * batchSize, acc[1], 0);
        float* devAbatch = hc::am_alloc(sizeof(float) * lenx * leny * batchSize, acc[1], 0);
        for(int i = 0;i < lenx * batchSize;i++) {
            xbatch[i] = rand() % 10;
        }
        for(int i = 0;i< lenx * leny * batchSize;i++) {
            Abatch[i] = rand() % 25;
        }
#ifdef PROFILE
        for(int iter=0; iter<10; iter++) {
#endif
        for(int i = 0;i < leny * batchSize;i++) {
            ybatch[i] = rand() % 15;
            ycblasbatch[i] = ybatch[i];
        }
        accl_view.copy(xbatch, devXbatch, lenx * batchSize * sizeof(float));
        accl_view.copy(ybatch, devYbatch, leny * batchSize * sizeof(float));
        accl_view.copy(Abatch, devAbatch, lenx * leny * batchSize * sizeof(float));
        status =  hc.hcblas_sgemv(accl_view, hcOrder, typeA, M, N, alpha, devAbatch, aOffset, A_batchOffset, lda, devXbatch, xOffset, X_batchOffset, incX, beta, devYbatch, yOffset, Y_batchOffset, incY, batchSize);
        accl_view.copy(devYbatch, ybatch, leny * batchSize * sizeof(float));
        lda = (hcOrder)? M : N;
        for(int i =0 ; i < batchSize; i++)
            cblas_sgemv( order, transa, M, N, alpha, Abatch + i * M * N, lda , xbatch + i * row, incX, beta, ycblasbatch + i * col, incY );
        for(int i =0; i < leny * batchSize; i ++){
            if (ybatch[i] != ycblasbatch[i]){
                ispassed = 0;
                cout <<" HCSGEMV[" << i<< "] " << ybatch[i] << " does not match with CBLASSGEMV[" << i <<"] "<< ycblasbatch[i] << endl;
                break;
            }
            else
                continue;
        }
        if(!ispassed) cout << "TEST FAILED" << endl; 
        if(status) cout << "TEST FAILED" << endl; 
#ifdef PROFILE
        }
#endif
        free(xbatch);
        free(ybatch);
        free(Abatch);
        free(ycblasbatch);
        hc::am_free(devAbatch);
        hc::am_free(devXbatch);
        hc::am_free(devYbatch);
   }
    return 0;
}
