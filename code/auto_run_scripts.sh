#!/bin/bash
for i in {1..40}
do
   mpirun -n 4 python3 parallelNystrom.py
done
