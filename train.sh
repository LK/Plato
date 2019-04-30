#!/bin/bash

N=$1
ROBOCODE=$2

: ${N:=2}
: ${ROBOCODE:="$HOME/robocode"}

start_bg () {
  cd /Users/Lenny/robocode/
  java -Xmx512M -Dsun.io.useCanonCaches=false -Ddebug=true -DNOSECURITY=true -DROBOTPATH=$ROBOCODE/robots -Dfile.encoding=UTF-8 -classpath "/Users/lenny/code/Plato/plato-robot/bin/lk/*:$ROBOCODE/libs/*:/Users/lenny/code/Plato/plato-robot/libs/*" robocode.Robocode -battle /Users/lenny/code/Plato/train.battle -tps 150 &
  sleep 1
  cd /Users/lenny/code/Plato/
}

rm -r /tmp/plato

javac -cp "plato-robot/libs/*:$ROBOCODE/libs/*" -Xlint:deprecation -d plato-robot/bin plato-robot/src/lk/*

tensorboard --logdir=/tmp/plato &
cd plato-server
python3 main.py &
cd ..

for i in $(seq 1 $N); do start_bg; done

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

wait