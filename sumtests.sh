#!/bin/bash 
# sumtests.sh [test_output_file]

sed 's/^.*//g' $1 > $1.filt

PASS=$(grep PASS $1.filt | wc -l)
TOTAL=$(grep '\.\.\.\.\.\.\.\.\.\.\.\.' $1.filt | wc -l)
echo PASS, $PASS
echo -n ERROR,  && grep ERROR $1.filt | wc -l
echo -n FAIL,  && grep FAIL $1.filt | wc -l
echo -n WAIT,  && grep WAIT $1.filt | wc -l
echo  Total,  $TOTAL
echo -n "PassRate, "  && echo "scale=2;$PASS * 100.0 / $TOTAL" | bc
