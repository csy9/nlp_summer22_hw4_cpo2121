#!/bin/bash

for f in *.predict
do
    echo ""
    echo "Scoring $f:"
    perl score.pl $f gold.trial
done
