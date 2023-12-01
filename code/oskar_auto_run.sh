#!/bin/bash
python3 setup_test.py <<EOF
B_matrix0_error_vs_k_vs_l
1
EOF

for i in {1..40}
do
   mpirun -n 4 python3 parallelNystrom.py
done

python3 setup_test.py <<EOF
B_matrix1_error_vs_k_vs_l
1
EOF

for i in {1..40}
do
   mpirun -n 4 python3 parallelNystrom.py
done

python3 setup_test.py <<EOF
B_matrix2_error_vs_k_vs_l
1
EOF

for i in {1..40}
do
   mpirun -n 4 python3 parallelNystrom.py
done

python3 setup_test.py <<EOF
B_matrix3a_error_vs_k_vs_l
1
EOF

for i in {1..40}
do
   mpirun -n 4 python3 parallelNystrom.py
done

python3 setup_test.py <<EOF
B_matrix3b_error_vs_k_vs_l
1
EOF

for i in {1..40}
do
   mpirun -n 4 python3 parallelNystrom.py
done

python3 setup_test.py <<EOF
B_seq_matrix2_error_vs_k_vs_l
1
EOF

for i in {1..40}
do
   python3 seqNystrom.py
done
