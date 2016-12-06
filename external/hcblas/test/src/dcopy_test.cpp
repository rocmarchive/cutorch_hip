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
    long xOffset = 0;
    int incY = 1;
    long yOffset = 0;
    hcblasStatus status;
    int batchSize = 32;
    long X_batchOffset = N;
    long Y_batchOffset = N;
    long lenx = 1 + (N-1) * abs(incX);
    long leny = 1 + (N-1) * abs(incY);
    /* CBLAS implementation */
    bool ispassed = 1;
    std::vector<hc::accelerator>acc = hc::accelerator::get_all();
    accelerator_view accl_view = (acc[1].get_default_view());

/* Implementation type I - Inputs and Outputs are HCC double array containers */
    
    if (Imple_type == 1) {
	double *X = (double*)calloc(lenx, sizeof(double));
	double *Y = (double*)calloc(leny, sizeof(double));
        double *Ycblas = (double*)calloc(leny, sizeof(double));
	double* devX = hc::am_alloc(sizeof(double) * lenx, acc[1], 0);
	double* devY = hc::am_alloc(sizeof(double) * leny, acc[1], 0);
        for(int i = 0;i < lenx;i++){
            X[i] = rand() % 10;
        }
        for(int i = 0;i < leny;i++){
            Y[i] =  rand() % 15;
            Ycblas[i] = Y[i];
        }
	accl_view.copy(X, devX, lenx * sizeof(double));
	accl_view.copy(Y, devY, leny * sizeof(double));
        status = hc.hcblas_dcopy(accl_view, N, devX, incX, xOffset, devY, incY, yOffset);
	accl_view.copy(devY, Y, leny * sizeof(double));
        cblas_dcopy( N, X, incX, Ycblas, incY );
        for(int i = 0; i < leny; i++){
            if (Y[i] != Ycblas[i]){
                ispassed = 0;
                cout <<" HCDCOPY[" << i<< "] " << Y[i] << " does not match with CBLASDCOPY[" << i <<"] "<< Ycblas[i] << endl;
                break;
            }
            else
                continue;
        }
        if(!ispassed) cout << "TEST FAILED" << endl; 
        if(status) cout << "TEST FAILED" << endl;
        free(X);
        free(Y);
        free(Ycblas);
        hc::am_free(devX);
        hc::am_free(devY);	
     }

/* Implementation type II - Inputs and Outputs are HCC double array containers with batch processing */

    else{
	double *Xbatch = (double*)calloc(lenx * batchSize, sizeof(double));
        double *Ybatch = (double*)calloc(leny * batchSize, sizeof(double));
        double *Ycblasbatch = (double*)calloc(leny * batchSize, sizeof(double));	
	double* devXbatch = hc::am_alloc(sizeof(double) * lenx * batchSize, acc[1], 0);
	double* devYbatch = hc::am_alloc(sizeof(double) * leny * batchSize, acc[1], 0);
        for(int i = 0;i < lenx * batchSize;i++){
            Xbatch[i] = rand() % 10;
        }
        for(int i = 0;i < leny * batchSize;i++){
            Ybatch[i] =  rand() % 15;
            Ycblasbatch[i] = Ybatch[i];
         }
	accl_view.copy(Xbatch, devXbatch, lenx * batchSize * sizeof(double));
	accl_view.copy(Ybatch, devYbatch, leny * batchSize * sizeof(double));
        status= hc.hcblas_dcopy(accl_view, N, devXbatch, incX, xOffset, devYbatch, incY, yOffset, X_batchOffset, Y_batchOffset, batchSize);
	accl_view.copy(devYbatch, Ybatch, leny * batchSize * sizeof(double));
        for(int i = 0; i < batchSize; i++)
        	cblas_dcopy( N, Xbatch + i * N, incX, Ycblasbatch + i * N, incY );
        for(int i =0; i < leny * batchSize; i++){
            if (Ybatch[i] != Ycblasbatch[i]){
                ispassed = 0;
                cout <<" HCDCOPY[" << i<< "] " <<Ybatch[i] << " does not match with CBLASDCOPY[" << i <<"] "<< Ycblasbatch[i] << endl;
                break;
            }
            else 
              continue;  
        }
        if(!ispassed) cout << "TEST FAILED" << endl; 
        if(status) cout << "TEST FAILED" << endl; 
	free(Xbatch);
	free(Ybatch);
	free(Ycblasbatch);
	hc::am_free(devXbatch);
	hc::am_free(devYbatch);
    }
    return 0;
}
