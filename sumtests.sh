
#!/bin/bash 
# sumtests.sh [test_output_file]

sed 's/.* //g' $1 > $1.filt

echo -n PASS,  && grep PASS $1.filt | wc -l
echo -n ERROR,  && grep ERROR $1.filt | wc -l
echo -n FAIL,  && grep FAIL $1.filt | wc -l
echo -n WAIT,  && grep WAIT $1.filt | wc -l
echo -n FAULT,  && grep fault $1.filt | wc -l
echo -n Total, && grep '\.\.\.\.\.\.\.\.\.\.\.\.' $1.filt | wc -l
