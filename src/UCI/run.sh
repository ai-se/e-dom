#! /bin/tcsh

rm out/*
rm err/*
rm log/*
rm dump/*

foreach VAR ("annealing" "audit" "autism" "bank" "bankrupt" "biodegrade" "blood-transfusion" "car" "cardiotocography" "cervical-cancer" "climate-sim" "contraceptive" "credit-approval" "credit-default" "crowdsource" "drug-consumption" "electric-stable" "gamma" "hand" "hepmass" "htru2" "image" "kddcup" "liver" "mushroom" "phishing" "sensorless-drive" "shop-intention")
  bsub -q standard -W 5000 -n 8 -o ./out/$VAR.out.%J -e ./err/$VAR.err.%J /share/tjmenzie/aagrawa8/miniconda2/bin/python2.7 main_d2h.py _test "$VAR" > log/"$VAR".log
end