#include <iostream>
#include "hcblaslib.h"
#include <cstdlib>
#include "cblas.h"
#include "hc_am.hpp"
using namespace std;

int main(int argc,char* argv[])
{  
    /* HCBLAS Implementation */
    hc::accelerator accl;
    Hcblaslibrary hc(&accl);  
    if (argc < 7) {
        cout<<"No sufficient commandline arguments specified"<<"argc :"<<argc<<endl;
        return -1;
    }
    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int K = atoi(argv[3]);
    int isTransA = (atoi(argv[4]));
    int isTransB = (atoi(argv[5])); 
    long lda = (atoi(argv[6]));
    long ldb = (atoi(argv[7]));
    long ldc = (atoi(argv[8]));
    float alpha = (atoi(argv[9]));
    float beta = (atoi(argv[10]));
    int incX = 1;
    int incY = 1;
    long aOffset = (atoi(argv[11]));
    long bOffset = (atoi(argv[12]));
    long cOffset = (atoi(argv[13]));
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

    float *C_cblas = (float*) calloc(M * N  + cOffset, sizeof(float));
    float *A = (float*) calloc(M * K  + aOffset, sizeof(float));
    float *B = (float*) calloc(K * N  + bOffset, sizeof(float));
    float *C = (float*) calloc(M * N  + cOffset, sizeof(float));
    float* devA = hc::am_alloc(sizeof(float) * (M * K + aOffset), acc[1], 0);
    float* devB = hc::am_alloc(sizeof(float) * (K * N + bOffset), acc[1], 0);
    float* devC = hc::am_alloc(sizeof(float) * (M * N + cOffset), acc[1], 0);
    for(int i = 0; i < M * K; i++) {
        A[i + aOffset] = rand()%100;
    }
    for(int i = 0; i < K * N;i++) {
        B[i  + bOffset] = rand() % 15;
    }
#ifdef PROFILE
    for(int iter = 0; iter < 10; iter++) {
#endif
        for(int i = 0; i < M * N;i++) {
            C[i  + cOffset] = rand() % 25;
            C_cblas[i + cOffset] = C[i + cOffset];
        }
        accl_view.copy(A, devA, (M * K + aOffset)* sizeof(float));
        accl_view.copy(B, devB, (K * N + bOffset)* sizeof(float));
        accl_view.copy(C, devC, (M * N + cOffset)* sizeof(float));
        status = hc.hcblas_sgemm(accl_view, hcOrder, typeA, typeB, M, N, K, alpha, devA, lda, devB, ldb, beta, devC, ldc, aOffset, bOffset, cOffset);
        accl_view.copy(devC, C,  (M * N + cOffset) * sizeof(float));
        cblas_sgemm( order, Transa, Transb, M, N, K, alpha, A + aOffset, lda, B + bOffset, ldb, beta, C_cblas + cOffset, ldc);
        for(int i = 0 ; i < M * N ; i++) { 
            if( C_cblas[i + cOffset] != (C[i + cOffset])) {
                 ispassed = 0;
                 cout << " HCSGEMM["<<i<<"] = "<<C[i + cOffset]<<" doesnot match with CBLASSGEMM["<<i<<"] =" << C_cblas[i + cOffset] << endl;
                 break;
            }
            else
                 continue;
        } 
        if(!ispassed) cout << "TEST FAILED" << endl; 
        if(status) cout << "TEST FAILED" << status<<  endl; 
#ifdef PROFILE
    }
#endif
    free(A);
    free(B);
    free(C);
    hc::am_free(devA);
    hc::am_free(devB);
    hc::am_free(devC);   
    return 0;   
}
   
   
  
