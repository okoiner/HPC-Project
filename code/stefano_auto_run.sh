#!/bin/bash
python3 setup_test.py <<EOF
C_matrix0_error_vs_k_vs_lratio
1
EOF

for i in {1..55}
do
   mpirun -n 4 python3 parallelNystrom.py
done

python3 setup_test.py <<EOF
C_matrix2_error_vs_k_vs_lratio
1
EOF

for i in {1..55}
do
   mpirun -n 4 python3 parallelNystrom.py
done

python3 setup_test.py <<EOF
C_matrix3a_error_vs_k_vs_lratio
1
EOF

for i in {1..55}
do
   mpirun -n 4 python3 parallelNystrom.py
done
