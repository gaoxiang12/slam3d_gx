#! /bin/sh

mkdir pcd ply rgb_index dep_index
python generateTxt.py
python associate.py rgb.txt dep.txt --max_difference 0.05 > temp.txt
python associate.py temp.txt odometry.txt --max_difference 0.05 > associate.txt
python change2index.py
python img2pcd.txt
rm -f temp.txt
