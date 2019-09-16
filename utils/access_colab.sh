#!/bin/bash

for i in `seq 0 12`
do
  echo "[$i]" ` date '+%y/%m/%d %H:%M:%S'` "connected."
  open 'https://colab.research.google.com/drive/1g1_hrfLFEAwYwGzUeMIivqxu9_szX29h#scrollTo=jysPsU8jt8tO'
  sleep 3600
done
