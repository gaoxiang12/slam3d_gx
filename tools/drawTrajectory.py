#! /usr/bin/python
import string
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

if __name__=='__main__':
    trajectory = file('trajectory.txt')
    Fig = plt.figure(1)
    x = []
    y = []
    for line in trajectory:
        data = line.split()
        x.append( string.atof(data[1]) )
        y.append( string.atof(data[3]) )
    plt.plot( x, y, 'ro-' )
    x = []
    y = []
    Fig.savefig("trajectory.pdf")
    Fig = plt.figure(2)
    odo = file('odometry.txt')
    for line in odo:
        data = line.split()
        x.append( -string.atof(data[2]))
        y.append( string.atof(data[1]))
    plt.plot( x, y, 'b--')
    Fig.savefig("odometry.pdf")
