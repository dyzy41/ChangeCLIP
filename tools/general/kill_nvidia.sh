#!/bin/bash

# 查找使用NVIDIA GPU的Python进程并杀死它们
for pid in $(fuser -v /dev/nvidia* 2>/dev/null | grep -Eo '[0-9]+'); do
    pname=$(ps -p $pid -o comm= 2>/dev/null)
    if [ "$pname" == "python" ]; then
        echo "Killing Python process with PID $pid"
        kill -9 $pid
    fi
done

