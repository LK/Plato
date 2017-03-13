#!/bin/bash

N=$1
ROBOCODE=$2

: ${N:=2}
: ${ROBOCODE:="$HOME/robocode"}

start_bg () {
  java -Xmx512M -Dsun.io.useCanonCaches=false -Ddebug=true -DNOSECURITY=true -DROBOTPATH=$ROBOCODE/robots -Dfile.encoding=UTF-8 -classpath "plato-robot/bin:$ROBOCODE/libs/*:plato-robot/libs/*" robocode.Robocode -battle ./train.battle -tps 200 > /dev/null &
}

rm -r /tmp/plato

tensorboard --logdir=/tmp/plato &
cd plato-server
python main.py &
cd ..

for i in $(seq 1 $N); do start_bg; done

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

wait