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
    int incX = 1;
    float dothcblas;
    long xOffset = 0;
    int incY = 1;
    long yOffset = 0;
    long X_batchOffset = N;
    long Y_batchOffset = N;
    int batchSize = 128;
    hcblasStatus status;
    if (N > 5000)
	batchSize = 50;
    /* CBLAS implementation */
    bool ispassed = 1;
    float  dotcblas = 0.0;
    float *dotcblastemp =(float*)calloc(batchSize, sizeof(float));
    /* CBLAS implementation */
    long lenx = 1 + (N-1) * abs(incX);
    long leny = 1 + (N-1) * abs(incY);
    std::vector<hc::accelerator>acc = hc::accelerator::get_all();
    accelerator_view accl_view = (acc[1].get_default_view());

/* Implementation type I - Inputs and Outputs are HCC float array containers */
    
    if (Imple_type == 1){
         float *X = (float*)calloc(lenx, sizeof(float));
         float *Y = (float*)calloc(leny, sizeof(float));
         float* devX = hc::am_alloc(sizeof(float) * lenx, acc[1], 0);
         float* devY = hc::am_alloc(sizeof(float) * leny, acc[1], 0);
         for(int i = 0;i < lenx;i++){
             X[i] = rand() % 10;
         }
        for(int i = 0;i < leny;i++){
             Y[i] = rand() % 15;
        }
        accl_view.copy(X, devX, lenx * sizeof(float));
        accl_view.copy(Y, devY, leny * sizeof(float));
        status = hc.hcblas_sdot(accl_view, N, devX, incX, xOffset, devY, incY, yOffset, dothcblas);
        dotcblas = cblas_sdot( N, X, incX, Y, incY);
        if (dothcblas != dotcblas){
            ispassed = 0;
            cout <<" HCSDOT " << dothcblas << " does not match with CBLASSDOT "<< dotcblas << endl;
        }
        if(!ispassed) cout << "TEST FAILED" << endl; 
        if(status) cout << "TEST FAILED" << endl;
        free(X);
        free(Y);
        hc::am_free(devX);
        hc::am_free(devY); 
     }

/* Implementation type II - Inputs and Outputs are HCC float array containers with batch processing */

    else{
        float *Xbatch = (float*)calloc(lenx * batchSize, sizeof(float));
        float *Ybatch = (float*)calloc(leny * batchSize, sizeof(float));
        float* devXbatch = hc::am_alloc(sizeof(float) * lenx * batchSize, acc[1], 0);
        float* devYbatch = hc::am_alloc(sizeof(float) * leny * batchSize, acc[1], 0);
        for(int i = 0;i < lenx * batchSize;i++){
            Xbatch[i] = rand() % 10;
        }
        for(int i = 0;i < leny * batchSize;i++){
            Ybatch[i] =  rand() % 15;
        }
        accl_view.copy(Xbatch, devXbatch, lenx * batchSize * sizeof(float));
        accl_view.copy(Ybatch, devYbatch, leny * batchSize * sizeof(float));
        status= hc.hcblas_sdot(accl_view, N, devXbatch, incX, xOffset, devYbatch, incY, yOffset, dothcblas, X_batchOffset, Y_batchOffset, batchSize);
        for(int i = 0; i < batchSize; i++){
        	dotcblastemp[i] = cblas_sdot( N, Xbatch + i * N, incX, Ybatch + i * N, incY);
                dotcblas += dotcblastemp[i];
        }
        if (dothcblas != dotcblas){
            ispassed = 0;
            cout <<" HCSDOT " << dothcblas << " does not match with CBLASSDOT "<< dotcblas << endl;
        }
        if(!ispassed) cout << "TEST FAILED" << endl; 
        if(status) cout << "TEST FAILED" << endl;
        free(Xbatch);
        free(Ybatch);
        hc::am_free(devXbatch);
        hc::am_free(devYbatch); 
    }
    return 0;
}
