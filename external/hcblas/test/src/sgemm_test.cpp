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
    int Imple_type = (atoi(argv[6])); 
    float alpha = 1;
    float beta = 1;
    long lda;
    long ldb;
    long ldc;
    int incX = 1;
    int incY = 1;
    long aOffset = 0;
    long bOffset = 0;
    long cOffset = 0;
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
            lda = (hcOrder)? M : K;
        }
        else {
            typeA = Trans;
            lda = (hcOrder)? K : M;
        }
        if(isTransB == 0) {
            typeB = NoTrans;
            ldb = (hcOrder)? K : N;
        }
        else {
            typeB = Trans;
            ldb = (hcOrder)? N : K;
        }
    }
    else {
        cout<< "Invalid Transpose type specified"<<endl;
        return -1;
    } 
    ldc = (hcOrder)? M : N;

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


/* Implementation type I - Inputs and Outputs are HCC float array containers */

       if(Imple_type == 1) {/* MULTIPLE GPU CALL */
	    float *C_cblas = (float*) calloc(M * N, sizeof(float));
	    float *A = (float*) calloc(M * K, sizeof(float));
            float *B = (float*) calloc(K * N, sizeof(float));
	    float *C = (float*) calloc(M * N, sizeof(float));
            float* devA = hc::am_alloc(sizeof(float) * M * K, acc[1], 0);
	    float* devB = hc::am_alloc(sizeof(float) * K * N, acc[1], 0);
	    float* devC = hc::am_alloc(sizeof(float) * M * N, acc[1], 0);
            for(int i = 0; i < M * K; i++) {
                A[i] = rand()%100;
            }
            for(int i = 0; i < K * N;i++) {
                B[i] = rand() % 15;
            }
#ifdef PROFILE
            for(int iter = 0; iter < 10; iter++) {
#endif
            for(int i = 0; i < M * N;i++) {
            C[i] = rand() % 25;
            C_cblas[i] = C[i];
            }
	    accl_view.copy(A, devA, M * K * sizeof(float));
	    accl_view.copy(B, devB, K * N * sizeof(float));
	    accl_view.copy(C, devC, M * N * sizeof(float));
            status = hc.hcblas_sgemm(accl_view, hcOrder, typeA, typeB, M, N, K, alpha, devA, lda, devB, ldb, beta, devC, ldc, aOffset, bOffset, cOffset);
	    accl_view.copy(devC, C,  M * N * sizeof(float));
            cblas_sgemm( order, Transa, Transb, M, N, K, alpha, A, lda, B, ldb, beta, C_cblas, ldc);
            for(int i = 0 ; i < M * N ; i++) { 
                if( C_cblas[i] != (C[i])) {
                    ispassed = 0;
                    cout << " HCSGEMM["<<i<<"] = "<<C[i]<<" doesnot match with CBLASSGEMM["<<i<<"] =" << C_cblas[i] << endl;
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
	    free(A);
	    free(B);
	    free(C);
            hc::am_free(devA);
	    hc::am_free(devB);
	    hc::am_free(devC);
        }
    
/* Implementation type II - Inputs and Outputs are HCC float array containers with batch processing */
 
        else {         
            float *Abatch = (float*) calloc(M * K, sizeof(float));
            float *Bbatch = (float*) calloc(K * N, sizeof(float));
            float *Cbatch = (float*) calloc(M * N * batchSize, sizeof(float));
            float *CCblasbatch = (float*) calloc(M * N * batchSize, sizeof(float));                    
	    float* devAbatch = hc::am_alloc(sizeof(float) * M * K, acc[1], 0);
	    float* devBbatch = hc::am_alloc(sizeof(float) * K * N, acc[1], 0);
	    float* devCbatch = hc::am_alloc(sizeof(float) * M * N * batchSize, acc[1], 0);

            for(int i = 0; i < M * K; i++) {
                Abatch[i] = rand()%100;
            }
            for(int i = 0; i < K * N;i++) {
                Bbatch[i] = rand() % 15;
            }
#ifdef PROFILE
            for(int iter = 0; iter < 10; iter++) {
#endif
            for(int i = 0; i < M * N * batchSize;i++) {
                Cbatch[i] = rand() % 25;
                CCblasbatch[i] = Cbatch[i];
            }
	    accl_view.copy(Abatch, devAbatch, M * K * sizeof(float));
	    accl_view.copy(Bbatch, devBbatch, K * N * sizeof(float));
	    accl_view.copy(Cbatch, devCbatch, M * N * batchSize * sizeof(float));
            status = hc.hcblas_sgemm(accl_view, hcOrder, typeA, typeB, M, N, K, alpha, devAbatch, lda, A_batchOffset, devBbatch, ldb, B_batchOffset, beta, devCbatch, ldc, C_batchOffset, aOffset, bOffset, cOffset, batchSize);   
	    accl_view.copy(devCbatch, Cbatch,  M * N * batchSize * sizeof(float));
            for(int i = 0; i < batchSize; i++)
                cblas_sgemm( order, Transa, Transb, M, N, K, alpha, Abatch, lda, Bbatch, ldb, beta, CCblasbatch  + i * M * N ,ldc );

            for(int i = 0 ; i < M * N * batchSize; i++){ 
                if( Cbatch[i] != (CCblasbatch[i])){
                    ispassed = 0;
                    cout << " HCSGEMM["<<i<<"] = "<<Cbatch[i]<<" doesnot match with CBLASSGEMM["<<i<<"] =" << CCblasbatch[i] << endl;
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
	    free(Abatch);
	    free(Bbatch);
	    free(Cbatch);
	    hc::am_free(devAbatch);
	    hc::am_free(devBbatch);
	    hc::am_free(devCbatch);
       } 
    return 0;   
}
   
   
  
