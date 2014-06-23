#! /usr/bin/python
import os
if __name__=='__main__':
    f = file("associate.txt")
    index = 1
    for s in f:
        rgb = s.split()[1]
        dep = s.split()[3]
        command = 'cp '+rgb+" rgb_index/"+str(index)+".png"
        print command
        os.system(command)
        command = 'cp '+dep+" dep_index/"+str(index)+".png"
        print command
        os.system(command)
        index = index+1
