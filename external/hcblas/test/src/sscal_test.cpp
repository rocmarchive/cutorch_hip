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
    long xOffset = 0;
    hcblasStatus status;
    int batchSize = 128;
    long X_batchOffset = N; 
    long lenx = 1 + (N-1) * abs(incX);
    std::vector<hc::accelerator>accs = hc::accelerator::get_all();
    accelerator_view accl_view = (accs[1].get_default_view());
    bool ispassed = 1;

/* Implementation type I - Inputs and Outputs are HCC device pointers */
    
    if (Imple_type == 1) {
	float *X = (float*)calloc(lenx, sizeof(float)); //host input
	float *Xcblas = (float*)calloc(lenx, sizeof(float));
        float* devX = hc::am_alloc(sizeof(float) * lenx, accs[1], 0);
        for(int i = 0;i < lenx;i++){
            X[i] = rand() % 10;
            Xcblas[i] = X[i];
        }
	accl_view.copy(X, devX, lenx * sizeof(float));
        status = hc.hcblas_sscal(accl_view, N, alpha, devX, incX, xOffset);
	accl_view.copy(devX, X, lenx * sizeof(float));
        cblas_sscal( N, alpha, Xcblas, incX );
        for(int i = 0; i < lenx ; i++){
            if (X[i] != Xcblas[i]){
                ispassed = 0;
                cout <<" HCSSCAL[" << i<< "] " << X[i] << " does not match with CBLASSSCAL[" << i <<"] "<< Xcblas[i] << endl;
                break;
            }
            else
                continue;
        }
        if(!ispassed) cout << "TEST FAILED" << endl; 
        if(status) cout << "TEST FAILED" << endl;
        hc::am_free(devX);
        free(X);
        free(Xcblas);	
   }

/* Implementation type II - Inputs and Outputs are HCC device pointers with batch processing */

    else{
	float *Xbatch = (float*)calloc(lenx * batchSize, sizeof(float));//host input
        float* devXbatch = hc::am_alloc(sizeof(float) * lenx * batchSize, accs[1], 0);
        float *Xcblasbatch = (float*)calloc(lenx * batchSize, sizeof(float));	
        for(int i = 0;i < lenx * batchSize;i++){
            Xbatch[i] = rand() % 10;
            Xcblasbatch[i] =  Xbatch[i];
         }
	accl_view.copy(Xbatch, devXbatch, lenx * batchSize * sizeof(float));
        status= hc.hcblas_sscal(accl_view, N, alpha, devXbatch, incX, xOffset, X_batchOffset, batchSize);
	accl_view.copy(devXbatch, Xbatch, lenx * batchSize * sizeof(float));
        for(int i = 0; i < batchSize; i++)
        	cblas_sscal( N, alpha, Xcblasbatch + i * N, incX);
        for(int i =0; i < lenx * batchSize; i ++){
            if (Xbatch[i] != Xcblasbatch[i]){
                ispassed = 0;
                cout <<" HCSSCAL[" << i<< "] " << Xbatch[i] << " does not match with CBLASSSCAL[" << i <<"] "<< Xcblasbatch[i] << endl;
                break;
            }
            else 
              continue;  
        }
        if(!ispassed) cout << "TEST FAILED" << endl; 
        if(status) cout << "TEST FAILED" << endl; 
	hc::am_free(devXbatch);
	free(Xbatch);
	free(Xcblasbatch);
    }
    return 0;
}
