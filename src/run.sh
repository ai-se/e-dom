#! /bin/tcsh

rm out/*
rm err/*
rm log/*
rm ../dump/*

foreach VAR ("0.025" "0.05" "0.1" "0.2")
  bsub -q standard -W 5000 -n 16 -o ./out/$VAR.out.%J -e ./err/$VAR.err.%J /share/tjmenzie/aagrawa8/miniconda2/bin/python2.7 main.py _test "$VAR" > log/"$VAR".log
end