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
    const double alpha = 1;
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
	double *X = (double*)calloc(lenx, sizeof(double)); //host input
	double *Xcblas = (double*)calloc(lenx, sizeof(double));
        double* devX = hc::am_alloc(sizeof(double) * lenx, accs[1], 0);
        for(int i = 0;i < lenx;i++){
            X[i] = rand() % 10;
            Xcblas[i] = X[i];
        }
	accl_view.copy(X, devX, lenx * sizeof(double));
        status = hc.hcblas_dscal(accl_view, N, alpha, devX, incX, xOffset);
	accl_view.copy(devX, X, lenx * sizeof(double));
        cblas_dscal( N, alpha, Xcblas, incX );
        for(int i = 0; i < lenx ; i++){
            if (X[i] != Xcblas[i]){
                ispassed = 0;
                cout <<" HCDSCAL[" << i<< "] " << X[i] << " does not match with CBLASDSCAL[" << i <<"] "<< Xcblas[i] << endl;
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
	double *Xbatch = (double*)calloc(lenx * batchSize, sizeof(double));//host input
        double* devXbatch = hc::am_alloc(sizeof(double) * lenx * batchSize, accs[1], 0);
        double *Xcblasbatch = (double*)calloc(lenx * batchSize, sizeof(double));	
        for(int i = 0;i < lenx * batchSize;i++){
            Xbatch[i] = rand() % 10;
            Xcblasbatch[i] =  Xbatch[i];
         }
	accl_view.copy(Xbatch, devXbatch, lenx * batchSize * sizeof(double));
        status= hc.hcblas_dscal(accl_view, N, alpha, devXbatch, incX, xOffset, X_batchOffset, batchSize);
	accl_view.copy(devXbatch, Xbatch, lenx * batchSize * sizeof(double));
        for(int i = 0; i < batchSize; i++)
        	cblas_dscal( N, alpha, Xcblasbatch + i * N, incX);
        for(int i =0; i < lenx * batchSize; i ++){
            if (Xbatch[i] != Xcblasbatch[i]){
                ispassed = 0;
                cout <<" HCDSCAL[" << i<< "] " << Xbatch[i] << " does not match with CBLASDSCAL[" << i <<"] "<< Xcblasbatch[i] << endl;
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
