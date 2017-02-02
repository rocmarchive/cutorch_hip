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
    double asumhcblas;
    int incX = 1;
    long xOffset = 0;
    hcblasStatus status;
    int batchSize = 128;
    long X_batchOffset = N;
    if(N > 10000)
	batchSize = 50;
    /* CBLAS implementation */
    bool ispassed = 1;
    double asumcblas = 0.0;
    double *asumcblastemp = (double*)calloc(batchSize, sizeof(double));
    /* CBLAS implementation */
    long lenx = 1 + (N-1) * abs(incX);
    std::vector<hc::accelerator>acc = hc::accelerator::get_all();
    accelerator_view accl_view = (acc[1].get_default_view());

/* Implementation type I - Inputs and Outputs are HCC double array containers */
      
    if (Imple_type == 1) {
        double *X = (double*)calloc(lenx, sizeof(double));
        double* devX = hc::am_alloc(sizeof(double) * lenx, acc[1], 0);
        for(int i = 0;i < lenx;i++){
            X[i] = rand() % 10;
        }
        accl_view.copy(X, devX, lenx * sizeof(double));
        status = hc.hcblas_dasum(accl_view, N, devX, incX, xOffset, &asumhcblas);
        asumcblas = cblas_dasum( N, X, incX);
        if (asumhcblas != asumcblas) {
            ispassed = 0;
            cout <<" HCDASUM " << asumhcblas << " does not match with CBLASDASUM "<< asumcblas << endl;
        }
        if(!ispassed) cout << "TEST FAILED" << endl; 
        if(status) cout << "TEST FAILED" << endl;
        free(X);
        hc::am_free(devX); 
     }

/* Implementation type II - Inputs and Outputs are HCC double array containers with batch processing */

    else{
        double *Xbatch = (double*)calloc(lenx * batchSize, sizeof(double));
        double *devXbatch = hc::am_alloc(sizeof(double) * lenx * batchSize, acc[1], 0);
        for(int i = 0;i < lenx * batchSize;i++) {
            Xbatch[i] = rand() % 10;
        }
        accl_view.copy(Xbatch, devXbatch, lenx * batchSize * sizeof(double));
        status= hc.hcblas_dasum(accl_view, N, devXbatch, incX, xOffset, &asumhcblas, X_batchOffset, batchSize);
        for(int i = 0; i < batchSize; i++) {
        	asumcblastemp[i] = cblas_dasum( N, Xbatch + i * N, incX);
                asumcblas += asumcblastemp[i];
        }
        if (asumhcblas != asumcblas) {
            ispassed = 0;
            cout <<" HCDASUM " << asumhcblas << " does not match with CBLASDASUM "<< asumcblas << endl;
        }
        if(!ispassed) cout << "TEST FAILED" << endl; 
        if(status) cout << "TEST FAILED" << endl; 
        free(Xbatch);
        hc::am_free(devXbatch);
    }
    return 0;
}
