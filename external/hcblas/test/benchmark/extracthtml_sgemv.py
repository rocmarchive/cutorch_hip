
# collecting data from the KernelSummary file and writing it to another file
from bs4 import BeautifulSoup
import sys
import csv
import os
filename=str(sys.argv[1])
Mval=str(sys.argv[2])
Nval=str(sys.argv[3])
transA=str(sys.argv[4])
Imple=str(sys.argv[5])
HCBLAS_PATH=str(os.environ['HCBLAS_PATH'])
inputfile=open(filename,"r")
out = csv.writer(open(HCBLAS_PATH + "/test/benchmark/profileSummary_sgemv.csv","a"), delimiter='\t',quoting=csv.QUOTE_NONE)
html=inputfile.read()

soup = BeautifulSoup(html, 'lxml')
table = soup.find("table", attrs={"class":"draggable sortable"})

# The first tr contains the field names.
headings = [th.get_text() for th in table.find("tr").find_all("th")]

datasets = []
for row in table.find_all("tr")[1:]:
    dataset = zip(headings, (td.get_text() for td in row.find_all("td")))
    datasets.append(dataset)

vlist=[]
#header =['M','N','K','TransA','TransB','Avg time']
#out.writerow(header)
for dataset in datasets:
    for field in dataset:
        if field[0].encode('ascii') == "Avg Time(ms)" :
           avgtime=field[1].encode('ascii')
           vlist = [Mval,Nval,transA,Imple,avgtime]
           out.writerow(vlist)
           vlist = []
