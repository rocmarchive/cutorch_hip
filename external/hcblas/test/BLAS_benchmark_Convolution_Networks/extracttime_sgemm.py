
# collecting data from the KernelSummary file and writing it to another file
from bs4 import BeautifulSoup
import sys
import csv
import os
filename=str(sys.argv[1])
Mval=str(sys.argv[2])
Nval=str(sys.argv[3])
Kval=str(sys.argv[4])
transA=str(sys.argv[5])
transB=str(sys.argv[6])
lda=str(sys.argv[7])
ldb=str(sys.argv[8])
ldc=str(sys.argv[9])
alpha=str(sys.argv[10])
beta=str(sys.argv[11])
aoff=str(sys.argv[12])
boff=str(sys.argv[13])
coff=str(sys.argv[14])
HCBLAS_PATH=str(os.environ['HCBLAS_PATH'])
inputfile=open(filename,"r")
out = csv.writer(open(HCBLAS_PATH +"/test/BLAS_benchmark_Convolution_Networks/Benchmark_sgemm.csv","a"), delimiter='\t',quoting=csv.QUOTE_NONE, quotechar='')
lines = inputfile.readlines()
avgtime = lines[0].split(":")[1].split("\n")[0]
gflops = lines[2].split(":")[1].split("\n")[0]
print avgtime
print gflops
vlist=[]
vlist = [Mval,Nval,Kval,transA,transB,lda,ldb,ldc,alpha,beta,aoff,boff,coff,avgtime,gflops]
out.writerow(vlist)
vlist = []
