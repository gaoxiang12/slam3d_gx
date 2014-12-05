#! /usr/bin/python

import os
import random

testset = []
test = 100
start = 1
end = 20
detect = "ORB"
descriptor = "SIFT"
os.system("rm data/exp1/error.log")

for i in range(0, test):
	r = random.randint(1, 2150)
	print r
	testset.append(r)

for t in testset:
	for i in range(start, end):
		os.system("bin/exp1_2 "+str(t)+" "+str(t+i)+" "+detect+" "+descriptor+" p")
os.system("mv data/exp1/error.log data/exp1/error_planar_"+detect+"_"+descriptor+".log")

for t in testset:
	for i in range(start, end):
		os.system("bin/exp1_2 "+str(t)+" "+str(t+i)+" "+detect+" "+descriptor+" n")
os.system("mv data/exp1/error.log data/exp1/error_normal_"+detect+"_"+descriptor+".log")
