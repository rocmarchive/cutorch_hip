#include <iostream>
#include "hcblaslib.h"
#include <cstdlib> 
#include "cblas.h"
#include "hc_am.hpp"

using namespace std;
int main(int argc, char** argv)
{   
    /* HCBLAS implementation */
    hc::accelerator accl;
    Hcblaslibrary hc(&accl);  
    if (argc < 3){
        cout<<"No sufficient commandline arguments specified"<<"argc :"<<argc<<endl;
        return -1;
    }
    int N = atoi(argv[1]);
    int Imple_type = atoi(argv[2]);
    const float alpha = 1;
    int incX = 1;
    int incY = 1;
    long xOffset = 0;
    long yOffset = 0;
    long X_batchOffset = N;
    long Y_batchOffset = N;
    int batchSize = 128;
    hcblasStatus status;
    /* CBLAS implementation */
    bool ispassed = 1;
    long lenx = 1 + (N-1) * abs(incX);
    long leny = 1 + (N-1) * abs(incY);
    std::vector<hc::accelerator>acc = hc::accelerator::get_all();
    accelerator_view accl_view = (acc[1].get_default_view());
 
/* Implementation type I - Inputs and Outputs are HCC float array containers */
    if(Imple_type == 1) {
        float *X = (float*)calloc(lenx, sizeof(float));
        float *Y = (float*)calloc(leny, sizeof(float));
        float *Ycblas = (float*)calloc(N, sizeof(float));
        float* devX = hc::am_alloc(sizeof(float) * lenx, acc[1], 0);
        float* devY = hc::am_alloc(sizeof(float) * leny, acc[1], 0);
        for(int i = 0;i < lenx;i++){
             X[i] = rand() % 10;
        }
#ifdef PROFILE
        for(int iter=0; iter<10; iter++) {
#endif
        for(int i = 0; i < leny; i++) {
             Y[i] =  rand() % 15;
             Ycblas[i] = Y[i];
        }
        accl_view.copy(X, devX, lenx * sizeof(float));
        accl_view.copy(Y, devY, leny * sizeof(float));
        status = hc.hcblas_saxpy(accl_view, N, alpha, devX, incX, devY, incY , xOffset, yOffset);
        accl_view.copy(devY, Y, leny * sizeof(float));
        cblas_saxpy( N, alpha, X, incX, Ycblas, incY );
        for(int i = 0; i < leny ; i++){
            if (Y[i] != Ycblas[i]){
                ispassed = 0;
                cout <<" HCSAXPY[" << i<< "] " << Y[i] << " does not match with CBLASSAXPY[" << i <<"] "<< Ycblas[i] << endl;
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
       free(X);
       free(Y);
       free(Ycblas);
       hc::am_free(devX);
       hc::am_free(devY);
     }

/* Implementation type II - Inputs and Outputs are HC++ float array containers with batch processing */

    else{
        float *Xbatch = (float*)calloc(lenx * batchSize, sizeof(float));
        float *Ybatch = (float*)calloc(leny * batchSize, sizeof(float));
        float *Ycblasbatch = (float*)calloc(N * batchSize, sizeof(float));
        float* devXbatch = hc::am_alloc(sizeof(float) * lenx * batchSize, acc[1], 0);
        float* devYbatch = hc::am_alloc(sizeof(float) * leny * batchSize, acc[1], 0);
        for(int i = 0;i < lenx * batchSize;i++){
            Xbatch[i] = rand() % 10;
         }
#ifdef PROFILE
        for(int iter = 0; iter < 10; iter++) {
#endif
        for(int i = 0;i < leny * batchSize;i++){
            Ybatch[i] =  rand() % 15;
            Ycblasbatch[i] = Ybatch[i];
        }
        accl_view.copy(Xbatch, devXbatch, lenx * batchSize * sizeof(float));
        accl_view.copy(Ybatch, devYbatch, leny * batchSize * sizeof(float));
        status= hc.hcblas_saxpy(accl_view, N, alpha, devXbatch, incX, X_batchOffset, devYbatch, incY, Y_batchOffset, xOffset, yOffset, batchSize);
        accl_view.copy(devYbatch, Ybatch, leny * batchSize * sizeof(float));
        for(int i = 0; i < batchSize; i++)
        	cblas_saxpy( N, alpha, Xbatch + i * N, incX, Ycblasbatch + i * N, incY );
        for(int i =0; i < leny * batchSize; i ++){
            if (Ybatch[i] != Ycblasbatch[i]){
                ispassed = 0;
                cout <<" HCSAXPY[" << i<< "] " << Ybatch[i] << " does not match with CBLASSAXPY[" << i <<"] "<< Ycblasbatch[i] << endl;
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
       free(Xbatch);
       free(Ybatch);
       free(Ycblasbatch);
       hc::am_free(devXbatch);
       hc::am_free(devYbatch);
    }
    return 0;
}
