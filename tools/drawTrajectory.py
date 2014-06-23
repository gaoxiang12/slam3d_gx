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
        x.append( string.atof(data[0]) )
        y.append( string.atof(data[2]) )
    plt.plot( x, y, 'ro-' )
    x = []
    y = []
    odo = file('odometry.txt')
    for line in odo:
        data = line.split()
        x.append( -string.atof(data[2]))
        y.append( string.atof(data[1]))
    plt.plot( x, y, 'b--')
    plt.legend(["estimated", "odometry"])
    plt.show()
    Fig.savefig("trajectory.pdf")
