#include<hc.hpp>
#include<iostream>
#include"hc_short_vector.hpp"
#include"hcblaslib.h"
#include<cblas.h>
#include<unistd.h>
#include"hc_am.hpp"
using namespace hc::short_vector;
using namespace hc;
using namespace std;
int main(int argc, char* argv[])
{
 /* HCBLAS implementation */
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
    long lda, ldb, ldc;
    long aOffset = 0;
    long bOffset = 0;
    long cOffset = 0;
    long A_batchOffset = 0;
    long B_batchOffset = 0;
    long C_batchOffset = M * N;
    int batchSize = 128;
    hcblasOrder hcOrder = ColMajor;
    hcblasStatus status; 
    hcblasTranspose typeA,typeB ;
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
    float_2 cAlpha, cBeta;
    cAlpha.x = 1;
    cAlpha.y = 1;
    cBeta.x = 1;
    cBeta.y = 1;
    /* CBLAS implementation */
    bool ispassed = 1;
    float alpha[2], beta[2];
    CBLAS_ORDER order = CblasColMajor;
    CBLAS_TRANSPOSE Transa, Transb;
    Transa = (typeA == NoTrans)? CblasNoTrans: CblasTrans;
    Transb = (typeB == NoTrans)? CblasNoTrans: CblasTrans;
    alpha[0] = cAlpha.x; 
    alpha[1] = cAlpha.y;
    beta[0] = cBeta.x;
    beta[1] = cBeta.y;
    std::vector<hc::accelerator>acc = hc::accelerator::get_all();
    hc::accelerator_view accl_view = (acc[1].get_default_view());
    if(M > 3000 && N > 3000) {
        batchSize = 25;
    }
    if(M > 9000 && N > 9000) {
        batchSize = 1;
    }

/* Implementation type I - Inputs and Outputs are HCC float array containers */

    if(Imple_type == 1) {
        float_2* a = (float_2 *)malloc(sizeof(float_2 )* M * K);
        float_2* b = (float_2 *)malloc(sizeof(float_2 )* K * N);
        float_2* c = (float_2 *)malloc(sizeof(float_2 )* M * N);
        float_2* devA = hc::am_alloc(sizeof(float_2) * M * K, acc[1], 0);
        float_2* devB = hc::am_alloc(sizeof(float_2) * K * N, acc[1], 0);
        float_2* devC = hc::am_alloc(sizeof(float_2) * M * N, acc[1], 0);
        float* ablas = (float *)malloc(sizeof(float )* M * K * 2);
        float* bblas = (float *)malloc(sizeof(float )* K * N * 2);
        float* cblas = (float *)malloc(sizeof(float )* M * N * 2); 
        int k = 0;
        for (int i = 0;i < M * K; i++) {
            a[i].x = rand() % 10;
            a[i].y = rand() % 20;
            ablas[k++] = a[i].x;
            ablas[k++] = a[i].y;
        }
        k = 0;
        for (int i = 0;i < K * N; i++) {
            b[i].x = rand() % 15;
            b[i].y = rand() % 25;
            bblas[k++] = b[i].x;
            bblas[k++] = b[i].y;
        }
#ifdef PROFILE
        for (int iter=0; iter<10; iter++) {
#endif
        k = 0;
        for (int i = 0;i < M * N; i++) {
            c[i].x = rand() % 18;
            c[i].y = rand() % 28;
            cblas[k++] = c[i].x;
            cblas[k++] = c[i].y;
        }
        accl_view.copy(a, devA, M * K * sizeof(float_2));
        accl_view.copy(b, devB, K * N * sizeof(float_2));
        accl_view.copy(c, devC, M * N * sizeof(float_2));
    	status = hc.hcblas_cgemm(accl_view, hcOrder, typeA, typeB, M, N, K, cAlpha, devA, aOffset, lda, devB, bOffset, ldb, cBeta, devC, cOffset, ldc);
        accl_view.copy(devC, c,  M * N * sizeof(float_2));
        cblas_cgemm( order, Transa, Transb, M, N, K, &alpha, ablas, lda, bblas, ldb, &beta, cblas, ldc );
        for(int i = 0,k = 0; ((i < M * N) && ( k < M * N * 2)) ; i++, k = k + 2){
            if ((c[i].x != cblas[k]) || (c[i].y != cblas[k+1])){
                ispassed = 0;
                cout <<" HCCGEMM_REAL[" << i<< "] " << c[i].x << " does not match with CBLASCGEMM_REAL[" << k <<"] "<< cblas[k] << endl;
                cout <<" HCCGEMM_IMG[" << i<< "] " << c[i].y << " does not match with CBLASCGEMM_IMG[" << k <<"] "<< cblas[k + 1] << endl;
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
        free(a);
        free(b);
        free(c);
        free(ablas);
        free(bblas);
        free(cblas);
        hc::am_free(devA);
        hc::am_free(devB);
        hc::am_free(devC);
    }

/* Implementation type II - Inputs and Outputs are HCC float array containers with batch processing */

    else{
        float_2 *Abatch = (float_2*) calloc(M * K, sizeof(float_2));
        float_2 *Bbatch = (float_2*) calloc(K * N, sizeof(float_2));
        float_2 *Cbatch = (float_2*) calloc(M * N * batchSize, sizeof(float_2));
        float_2* devAbatch = hc::am_alloc(sizeof(float_2) * M * K, acc[1], 0);
        float_2* devBbatch = hc::am_alloc(sizeof(float_2) * K * N, acc[1], 0);
        float_2* devCbatch = hc::am_alloc(sizeof(float_2) * M * N * batchSize, acc[1], 0);
        float* abatch = (float *)malloc(sizeof(float )* M * K * 2);
        float* bbatch = (float *)malloc(sizeof(float )* K * N * 2);
        float* cbatch = (float *)malloc(sizeof(float )* M * N * 2 * batchSize);
        int k = 0;
        for (int i = 0;i < M * K; i++) {
           Abatch[i].x = rand() % 10;
           Abatch[i].y = rand() % 20;
           abatch[k++] = Abatch[i].x;
           abatch[k++] = Abatch[i].y;
        }

        k = 0;
        for (int i = 0;i < K * N; i++) {
           Bbatch[i].x = rand() % 15;
           Bbatch[i].y = rand() % 25;
           bbatch[k++] = Bbatch[i].x;
           bbatch[k++] = Bbatch[i].y;
        }
#ifdef PROFILE
        for(int iter=0; iter<10; iter++) {   
#endif
        k = 0;
        for (int i = 0;i < M * N * batchSize; i++) {
           Cbatch[i].x = rand() % 18;
           Cbatch[i].y = rand() % 28;
           cbatch[k++] = Cbatch[i].x ;
           cbatch[k++] = Cbatch[i].y;
        } 
        accl_view.copy(Abatch, devAbatch, M * K * sizeof(float_2));
        accl_view.copy(Bbatch, devBbatch, K * N * sizeof(float_2));
        accl_view.copy(Cbatch, devCbatch, M * N * batchSize * sizeof(float_2));
    	status = hc.hcblas_cgemm(accl_view, hcOrder, typeA, typeB, M, N, K, cAlpha, devAbatch, aOffset, A_batchOffset, lda, devBbatch, bOffset, B_batchOffset, ldb, cBeta, devCbatch, cOffset, C_batchOffset, ldc, batchSize);
        accl_view.copy(devCbatch, Cbatch,  M * N * batchSize * sizeof(float_2)); 
        for(int i = 0; i < batchSize;i++)
	     cblas_cgemm( order, Transa, Transb, M, N, K, &alpha, abatch, lda, bbatch, ldb, &beta, cbatch + i * M * N * 2, ldc );
        for(int i = 0,k = 0; ((i < M * N * batchSize)&&( k < M * N * 2 * batchSize)); i++, k = k + 2){
            if ((Cbatch[i].x != cbatch[k]) || (Cbatch[i].y != cbatch[k+1])){
                ispassed = 0;
                cout <<" HCCGEMM_REAL[" << i<< "] " << Cbatch[i].x << " does not match with CBLASCGEMM_REAL[" << k <<"] "<< cbatch[k] << endl;
                cout <<" HCCGEMM_IMG[" << i<< "] " << Cbatch[i].y << " does not match with CBLASCGEMM_IMG[" << k <<"] "<< cbatch[k + 1] << endl;
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
        free(abatch);
        free(bbatch);
        free(cbatch);
        free(Abatch);
        free(Bbatch);
        free(Cbatch);
        hc::am_free(devAbatch);
        hc::am_free(devBbatch);
        hc::am_free(devCbatch);
    }
    return 0;
}

