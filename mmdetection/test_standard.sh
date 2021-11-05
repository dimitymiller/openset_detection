#!/bin/bash 

echo 'Testing with base weights:' $1 
echo 'Save name:' $2 
echo 'Dataset:' $3 


echo Testing data
if [ $3 == coco ]
then
    python test_data.py --subset train --dir "${1}${NUM}" --saveNm "${2}${NUM}" --dataset $3
else 
    python test_data.py --subset train07 --dir "${1}${NUM}" --saveNm "${2}${NUM}" --dataset $3
    python test_data.py --subset train12 --dir "${1}${NUM}" --saveNm "${2}${NUM}" --dataset $3
    
fi

python test_data.py --subset val --dir "${1}${NUM}" --saveNm "${2}${NUM}" --dataset $3
python test_data.py --subset test --dir "${1}${NUM}" --saveNm "${2}${NUM}" --dataset $3

echo Associating data
python associate_data.py FRCNN --saveNm "${2}" --dataset $3

echo Getting Results
python get_results.py FRCNN --saveNm "${2}" --dataset $3 --saveResults True
