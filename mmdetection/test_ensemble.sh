#!/bin/bash 

echo 'Testing ensembles with base weights:' $1 
echo 'Save names:' $2 
echo 'Dataset:' $3 

for NUM in 0 1 2 3 4
do
    echo Testing ensemble number $NUM
    if [ $3 == coco ]
    then
        python test_data.py --subset train --dir "${1}${NUM}" --saveNm "${2}${NUM}" --dataset $3
    else 
        python test_data.py --subset train07 --dir "${1}${NUM}" --saveNm "${2}${NUM}" --dataset $3
        python test_data.py --subset train12 --dir "${1}${NUM}" --saveNm "${2}${NUM}" --dataset $3
        
    fi

    python test_data.py --subset val --dir "${1}${NUM}" --saveNm "${2}${NUM}" --dataset $3
    python test_data.py --subset test --dir "${1}${NUM}" --saveNm "${2}${NUM}" --dataset $3
done

echo Merging detections
if [ $3 == coco ]
then
    python merge_ensemble.py --subset train --saveNm "${2}" --dataset $3
else 
    python merge_ensemble.py --subset train07 --saveNm "${2}" --dataset $3
    python merge_ensemble.py --subset train12 --saveNm "${2}" --dataset $3
    
fi

python merge_ensemble.py --subset val --saveNm "${2}" --dataset $3
python merge_ensemble.py --subset test --saveNm "${2}" --dataset $3

