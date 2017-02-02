#ifndef HELPER_FUNC_H
#define HELPER_FUNC_H

#include <iostream>
#include<vector>
#include<numeric>
#include <math.h>

float sgemmCompareL2fe(const float *reference, const float *data,
               const unsigned int len, const float epsilon)
{
    if(epsilon < 0) {
      std::cout<<"Invalid epsilon value"<<std::endl;
    }

    float error = 0;
    float ref = 0;

    // Making Vector conversions to invoke standar library routines
    std::vector<float> refVec(reference, reference + len);
    std::vector<float> dataVec(data, data + len);
    std::vector<float> diffVec(len);
    // Compute the difference vector with respect to reference data
    std::transform(refVec.begin(), refVec.end(), dataVec.begin(), diffVec.begin(), std::minus<float>());
    // Compute the square of the difference element wise
    std::transform(diffVec.begin(), diffVec.end(), diffVec.begin(), diffVec.begin(), std::multiplies<float>());
    error = std::inner_product(diffVec.begin(), diffVec.end(), diffVec.begin(), error);
    ref = std::inner_product(refVec.begin(), refVec.end(), refVec.begin(), ref);
    float normRef = sqrtf(ref);

    if (fabs(ref) < 1e-7)
    {
#ifdef _DEBUG
        std::cerr << "ERROR, reference l2-norm is 0\n";
#endif
        return false;
    }

    float normError = sqrtf(error);
    error = normError / normRef;
    bool result = error < epsilon;
#ifdef _DEBUG

    if (! result)
    {
        std::cerr << "ERROR, l2-norm error "
                  << error << " is greater than epsilon " << epsilon << "\n";
    }
#endif

      return error;
}

void printDiff(float *data1, float *data2, int width, int height, int iListLength, float fListTol)
{
    printf("Listing first %d Differences > %.6f...\n", iListLength, fListTol);
    int i,j,k;
    int error_count=0;

    for (j = 0; j < height; j++)
    {
        if (error_count < iListLength)
        {
            printf("\n  Row %d:\n", j);
        }

        for (i = 0; i < width; i++)
        {
            k = j * width + i;
            float fDiff = fabs(data1[k] - data2[k]);

            if (fDiff > fListTol)
            {
                if (error_count < iListLength)
                {
                    printf("    Loc(%d,%d)\tCPU=%.5f\tGPU=%.5f\tDiff=%.6f\n", i, j, data1[k], data2[k], fDiff);
                }

                error_count++;
            }
        }
    }

    printf(" \n  Total Errors = %d\n", error_count);
}
#endif
