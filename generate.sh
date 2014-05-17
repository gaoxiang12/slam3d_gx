#!/bin/sh
# build the template project

clear
cd build
cmake DCMAKE_BUILD_TYPE = Debug ..
make 2> error.log
cat error.log

