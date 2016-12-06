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
    if (argc < 4){
        cout<<"No sufficient commandline arguments specified"<<"argc :"<<argc<<endl;
        return -1;
    }
    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int Imple_type = atoi(argv[3]);
    float alpha = 1;
    long lda;
    int incX = 1;
    int incY = 1;
    long xOffset = 0;
    long yOffset = 0;
    long aOffset = 0;
    long X_batchOffset = M;
    long Y_batchOffset = N;
    long A_batchOffset = M * N;
    int batchSize = 128;
    long lenx,  leny; 
    hcblasStatus status;
    hcblasOrder hcOrder = ColMajor;
    lda = (hcOrder)? M : N;  
    lenx =  1 + (M-1) * abs(incX);
    leny =  1 + (N-1) * abs(incY);
    /* CBLAS implementation */
    bool ispassed = 1;
    CBLAS_ORDER order;
    order = CblasColMajor;
    std::vector<hc::accelerator>acc = hc::accelerator::get_all();
    accelerator_view accl_view = (acc[1].get_default_view());

/* Implementation type I - Inputs and Outputs are HCC float array containers */

        if(Imple_type == 1) {
            float *Acblas = (float *)calloc( lenx * leny , sizeof(float));
            float *x = (float*)calloc( lenx , sizeof(float));
            float *y = (float*)calloc( leny , sizeof(float));
            float *A = (float *)calloc( lenx * leny , sizeof(float));
            float* devA = hc::am_alloc(sizeof(float) * lenx * leny, acc[1], 0);
            float* devX = hc::am_alloc(sizeof(float) * lenx, acc[1], 0);
            float* devY = hc::am_alloc(sizeof(float) * leny, acc[1], 0);
            for(int i = 0;i < lenx;i++) {
                x[i] = rand() % 10;
            }
            for(int i = 0;i < leny;i++) {
                y[i] = rand() % 15;
            }
#ifdef PROFILE
            for(int iter=0; iter<10; iter++) {
#endif
            for(int i = 0;i< lenx * leny ;i++) {
                A[i] = rand() % 25;
                Acblas[i] = A[i];
            }
            accl_view.copy(A, devA, lenx * leny * sizeof(float));
            accl_view.copy(x, devX, lenx * sizeof(float));
            accl_view.copy(y, devY, leny * sizeof(float));
            status = hc.hcblas_sger(accl_view, hcOrder, M , N , alpha, devX, xOffset, incX, devY, yOffset, incY, devA, aOffset, lda );
            accl_view.copy(devA, A, lenx * leny * sizeof(float));
            cblas_sger( order, M, N, alpha, x, incX, y, incY, Acblas, lda);
            for(int i =0; i < lenx * leny ; i++){
                if (A[i] != Acblas[i]){
                    ispassed = 0;
                    cout <<" HCSGER[" << i<< "] " << A[i] << " does not match with CBLASSGER[" << i <<"] "<< Acblas[i] << endl;
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
            free(Acblas);
            hc::am_free(devA);
            hc::am_free(devX);
            hc::am_free(devY);
        }

/* Implementation type II - Inputs and Outputs are HCC float array containers with batch processing */

        else{
            float *xbatch = (float*)calloc( lenx * batchSize, sizeof(float));
            float *ybatch = (float*)calloc( leny * batchSize, sizeof(float));
            float *Abatch = (float *)calloc( lenx * leny * batchSize, sizeof(float));
            float *Acblasbatch = (float *)calloc( lenx * leny * batchSize, sizeof(float));
            float* devXbatch = hc::am_alloc(sizeof(float) * lenx * batchSize, acc[1], 0);
            float* devYbatch = hc::am_alloc(sizeof(float) * leny * batchSize, acc[1], 0);
            float* devAbatch = hc::am_alloc(sizeof(float) * lenx * leny * batchSize, acc[1], 0);
            for(int i = 0;i < lenx * batchSize;i++){
                xbatch[i] = rand() % 10;
            }
            for(int i = 0;i < leny * batchSize;i++){
                ybatch[i] = rand() % 15;
            }
#ifdef PROFILE
            for(int iter = 0; iter < 10; iter++) {
#endif
            for(int i = 0;i< lenx * leny * batchSize;i++){
                Abatch[i] = rand() % 25;
                Acblasbatch[i] = Abatch[i];
            }
            accl_view.copy(xbatch, devXbatch, lenx * batchSize * sizeof(float));
            accl_view.copy(ybatch, devYbatch, leny * batchSize * sizeof(float));
            accl_view.copy(Abatch, devAbatch, lenx * leny * batchSize * sizeof(float));
            status = hc.hcblas_sger(accl_view, hcOrder, M , N , alpha, devXbatch, xOffset, X_batchOffset, incX, devYbatch, yOffset, Y_batchOffset, incY, devAbatch, aOffset, A_batchOffset, lda, batchSize );
            accl_view.copy(devAbatch, Abatch, lenx * leny * batchSize * sizeof(float));
            for(int i = 0; i < batchSize; i++)
               cblas_sger( order, M, N, alpha, xbatch + i * M, incX, ybatch + i * N, incY, Acblasbatch + i * M * N, lda); 
            for(int i =0; i < lenx * leny * batchSize; i++){
               if (Abatch[i] != Acblasbatch[i]){
                   ispassed = 0;
                   cout <<" HCSGER[" << i<< "] " << Abatch[i] << " does not match with CBLASSGER[" << i <<"] "<< Acblasbatch[i] << endl;
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
        free(Acblasbatch);
        hc::am_free(devAbatch);
        hc::am_free(devXbatch);
        hc::am_free(devYbatch);
      }
    return 0;
}
