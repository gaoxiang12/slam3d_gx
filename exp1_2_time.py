#! /usr/bin/python

import os
import random

testset = []
test = 10
start = 1
end = 10
detect = "GridFAST"
descriptor = "SIFT"
os.system("rm data/time.log")

def fun():
        testset=[]
        for i in range(0, test):
                r = random.randint(1, 2150)
                testset.append(r)
        for t in testset:
                for i in range(start, end):
                        os.system("bin/exp1_2 "+str(t)+" "+str(t+i)+" "+detect+" "+descriptor+" p")
        os.system("mv data/time.log data/time_"+detect+"_"+descriptor+".log")

if __name__ == '__main__':
        detect = "GridFAST"
        fun()
        detect = "SIFT"
        fun()
        detect = "STAR"
        fun()
        detect = "ORB"
        fun()
        detect = "GFTT"
        fun()
        detect = 'SURF'
        descriptor = "SURF"
        fun()
        
