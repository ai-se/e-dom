#! /bin/tcsh

rm out/*
rm err/*
rm log/*
rm ../dump/*

foreach VAR ("camel" "ivy" "jedit" "log4j" "poi" "synapse" "velocity" "xalan" "xerces" "lucene")
  bsub -q standard -W 5000 -n 8 -o ./out/$VAR.out.%J -e ./err/$VAR.err.%J /share/tjmenzie/aagrawa8/miniconda2/bin/python2.7 main.py _test "$VAR" > log/"$VAR".log
end