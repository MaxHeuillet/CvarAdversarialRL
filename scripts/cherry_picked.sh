#!/bin/bash

for s in 1 2 3 4 5 6 7 8 9 10

do

	python3 cvarRL/evaluate_cherry_pick.py --model experiment3_100_0.1_0.7_1 --cost 0.7 --stochasticity 0.05 --type 5m --episodes 1 --budget 100 --seed $s 

	python3 cvarRL/evaluate_cherry_pick.py --model experiment3_25_0.1_0.7_1 --cost 0.7 --stochasticity 0.05 --type 5m --episodes 1 --budget 25 --seed $s 

#	python3 cvarRL/evaluate_cherry_pick.py --model experiment2_1_0.05_0.7_1 --cost 0.7 --stochasticity 0.05 --type 5m --episodes 1 --budget 1 --seed $s 



done
