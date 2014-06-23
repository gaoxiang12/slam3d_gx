#! /usr/bin/python

import os
import os.path

if __name__ == '__main__':
    rgbfile = open('rgb.txt', 'w')
    depfile = open('dep.txt', 'w')
    for parent, dirnames, filenames in os.walk('./rgb/'):
        for filename in filenames:
            time = filename[0:len(filename)-4]
            rgbfile.write(time+" rgb/"+time+".png\n")

    rgbfile.close()
    for parent, dirnames, filenames in os.walk('./dep/'):
        for filename in filenames:
            time = filename[0:len(filename)-4]
            depfile.write(time+" dep/"+time+".png\n")
    depfile.close()

    print 'txt file generated.'
