#! /bin/tcsh

rm out/*
rm err/*
rm log/*
rm ../dump/*

foreach VAR ("ant" "arc" "camel" "ivy" "jedit" "log4j" "poi" "prop" "redaktor" "synapse" "tomcat" "velocity" "xalan" "xerces")
  bsub -q standard -W 5000 -n 16 -o ./out/$VAR.out.%J -e ./err/$VAR.err.%J /share/tjmenzie/aagrawa8/miniconda2/bin/python2.7 main_popt.py _test "$VAR" > log/"$VAR".log
end