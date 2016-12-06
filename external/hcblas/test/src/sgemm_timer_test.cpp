#include <iostream>
#include "../include/helper_functions.h"
#include "hcblaslib.h"
#include <cstdlib>
#include "cblas.h"
#include "hc_am.hpp"
#include <chrono>
#include <vector>
#include <thread>

using namespace std;

template <typename T>
  T average(const std::vector<std::chrono::duration<T>> &data) {
  T avg_duration = 0;
  for(auto &i : data)
    avg_duration += i.count();
  return avg_duration/data.size();
}

double gflops( double Time, int M, int N, int K )
{
        return ((2.0 * M * N * K)/Time);
}

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
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    std::vector<std::chrono::duration<double>> elapsed_pfe;

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
    for(int iter = 0; iter < 10; iter++) {
        for(int i = 0; i < M * N;i++) {
            C[i  + cOffset] = rand() % 25;
            C_cblas[i + cOffset] = C[i + cOffset];
        }
        accl_view.copy(A, devA, (M * K + aOffset)* sizeof(float));
        accl_view.copy(B, devB, (K * N + bOffset)* sizeof(float));
        accl_view.copy(C, devC, (M * N + cOffset)* sizeof(float));
        start = std::chrono::high_resolution_clock::now();
        status = hc.hcblas_sgemm(accl_view, hcOrder, typeA, typeB, M, N, K, alpha, devA, lda, devB, ldb, beta, devC, ldc, aOffset, bOffset, cOffset);
        //accl_view.wait();
        end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> dur = end - start;
        elapsed_pfe.push_back(dur);
        accl_view.copy(devC, C,  (M * N + cOffset) * sizeof(float));
#if 1
        bool result = 0;
        float epsilon = 1.0e-5f;
        cblas_sgemm( order, Transa, Transb, M, N, K, alpha, A + aOffset, lda, B + bOffset, ldb, beta, C_cblas + cOffset, ldc);
        float err = sgemmCompareL2fe(C, C_cblas, M*N, epsilon);
        if (err > epsilon) {
          result = 1;
          std::cout << "Precison Error limit exceeded" << std::endl;
          printDiff(C_cblas, C, M, N, 100, 1.0e-5f);
          return -1;
        }

        if(result) cout << "TEST FAILED" << endl; 
#endif
        if(status) cout << "TEST FAILED" << status<<  endl; 
    }
    double Avg_time = average(elapsed_pfe);
    double time_in_ns = Avg_time * 1e9;
    double time_in_ms = Avg_time * 1e3;
    double gflopspersec = gflops(time_in_ns, M, N, K);
    cout << "BLAS Kernel execution time <ms>:" << time_in_ms <<endl;
    cout << "BLAS Kernel execution time <ns>:" << time_in_ns <<endl;
    cout << "BLAS kernel execution Gflops < 2.0*M*N*K/time >:" << gflopspersec <<endl;
    free(A);
    free(B);
    free(C);
    hc::am_free(devA);
    hc::am_free(devB);
    hc::am_free(devC);   
    return 0;   
}
   
   
  
