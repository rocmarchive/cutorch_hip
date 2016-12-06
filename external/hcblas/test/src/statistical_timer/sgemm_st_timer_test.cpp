#include <iostream>
#include <cstdlib>
#include <vector>
#include <thread>
#include <sys/types.h>
#include <stdio.h>
#include <cfloat>
#include <sys/time.h>
#include "statisticalTimer.h"
#include "hcblaslib.h"
#include "hc_am.hpp"
#include "cblas.h"
using namespace std;

int main(int argc,char* argv[])
{  
    /* HCBLAS Implementation */
    hc::accelerator accl;
    Hcblaslibrary hc(&accl); 
    if (argc < 3) {
        cout<<"No sufficient commandline arguments specified"<<"argc :"<<argc<<endl;
        return -1;
    }
    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int K = atoi(argv[3]);
    int isTransA = 0; //(atoi(argv[4]));
    int isTransB = 0; //(atoi(argv[5])); 
    long lda = M; //(atoi(argv[6]));
    long ldb = K; //(atoi(argv[7]));
    long ldc = M; //(atoi(argv[8]));
    float alpha = 1; //(atoi(argv[9]));
    float beta = 1; //(atoi(argv[10]));
    int incX = 1;
    int incY = 1;
    long aOffset = 0; //(atoi(argv[11]));
    long bOffset = 0; //(atoi(argv[12]));
    long cOffset = 0; //(atoi(argv[13]));
    long A_batchOffset = 0;
    long B_batchOffset = 0;
    long C_batchOffset = M * N;
    int batchSize = 128;
    hcblasOrder hcOrder = ColMajor;
    hcblasTranspose typeA, typeB;
    hcblasStatus status;
    if((isTransA == 0 || isTransA == 1) && (isTransB == 0 || isTransB == 1)) {
        if(isTransA == 0) {
            typeA = NoTrans;
        }
        else {
            typeA = Trans;
        }
        if(isTransB == 0) {
            typeB = NoTrans;
        }
        else {
            typeB = Trans;
        }
    }
    else {
        cout<< "Invalid Transpose type specified"<<endl;
        return -1;
    } 

    /* CBLAS implementation */
    bool ispassed = 1;
    CBLAS_ORDER order;
    CBLAS_TRANSPOSE Transa, Transb;
    order = CblasColMajor;
    Transa = (typeA == NoTrans)?CblasNoTrans:CblasTrans;
    Transb = (typeB == NoTrans)?CblasNoTrans:CblasTrans;
    if(M > 3000 && N > 3000){
	batchSize = 25;
    }
    if(M > 9000 && N > 9000){
        batchSize = 1;
    }
    std::vector<hc::accelerator>acc = hc::accelerator::get_all();
    accelerator_view accl_view = (acc[1].get_default_view()); 
    std::vector<double> elapsed_pfe;

    float *C_cblas = (float*) calloc(M * N * 10  + cOffset, sizeof(float));
    float *A = (float*) calloc(M * K  + aOffset, sizeof(float));
    float *B = (float*) calloc(K * N  + bOffset, sizeof(float));
    float *C = (float*) calloc(M * N * 10 + cOffset, sizeof(float));
    float* devA = hc::am_alloc(sizeof(float) * (M * K + aOffset), acc[1], 0);
    float* devB = hc::am_alloc(sizeof(float) * (K * N + bOffset), acc[1], 0);
    float* devC = hc::am_alloc(sizeof(float) * (M * N * 10 + cOffset), acc[1], 0);
    for(int i = 0; i < M * K; i++) {
        A[i + aOffset] = rand()%100;
    }
    for(int i = 0; i < K * N;i++) {
        B[i  + bOffset] = rand() % 15;
    }

    for(int i = 0; i < M * N * 10;i++) {
        C[i  + cOffset] = rand() % 25;
        C_cblas[i + cOffset] = C[i + cOffset];
    }
 
    accl_view.copy(devA, A, (M * K + aOffset)* sizeof(float));
    accl_view.copy(devB, B, (K * N + bOffset)* sizeof(float));
    accl_view.copy(devC, C, (M * N * 10 + cOffset)* sizeof(float));

    StatisticalTimer& timer = StatisticalTimer::getInstance( );
    StatisticalTimer::sTimerID timer_id;
    timer.Reserve( 3 , 20);
    timer.setNormalize( true );
    timer_id = timer.getUniqueID("st_sgemm", 0);

    timer.Start(timer_id);

    for(int iter = 0; iter < 10; iter++) {
      
       status = hc.hcblas_sgemm(accl_view, hcOrder, typeA, typeB, M, N, K, alpha, devA, lda, devB, ldb, beta, devC + (iter * M * N), ldc, aOffset, bOffset, cOffset);

#if 0
       accl_view.copy(C + (iter * M * N), devC + (iter * M * N),  (M * N + cOffset) * sizeof(float));

        cblas_sgemm( order, Transa, Transb, M, N, K, alpha, A + aOffset, lda, B + bOffset, ldb, beta, C_cblas + (iter * M * N)+ cOffset, ldc);
        for(int i = 0 ; i < M * N ; i++) { 
            if( C_cblas[i + (iter * M * N) + cOffset] != (C[i + (iter * M * N) + cOffset])) {
                 ispassed = 0;
                 cout << " HCSGEMM["<<i<<"] = "<<C[i + (iter * M * N) + cOffset]<<" doesnot match with CBLASSGEMM["<<i<<"] =" << C_cblas[i + (iter * M * N) + cOffset] << endl;
                 break;
            }
            else
                 continue;
        } 
        if(!ispassed) cout << "TEST FAILED" << endl; 
       if(status) cout << "TEST FAILED" << status<<  endl; 
#endif
     }

     timer.Stop(timer_id);

    accl_view.copy(C, devC,  (M * N * 10 + cOffset) * sizeof(float));

    double Avg_time = timer.getAverageTime(timer_id);
    double time_in_ns=Avg_time * 1e9;
    double time_in_ms = Avg_time * 1e3;
    double gflops = timer.gflops(time_in_ns/10, M, N, K);
    timer.pruneOutliers( 3.0 );
    cout << "BLAS Kernel execution time <ms>:" << time_in_ms/10<<endl;
    cout << "BLAS Kernel execution time <ns>:" << time_in_ns/10<<endl;
    cout << "BLAS kernel execution Gflops < 2.0*M*N*K/time >:" << gflops <<endl;

    free(A);
    free(B);
    free(C);
    hc::am_free(devA);
    hc::am_free(devB);
    hc::am_free(devC);   
    return 0;   
}
   
   
  
