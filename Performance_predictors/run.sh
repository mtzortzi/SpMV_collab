for implementation in AOCL CSR5 Vec-CSR Merge-CSR MKL-IE Naive-CSR SELL-C-s SparseX
do
    clear && python3 main.py -m mlp -s AMD-EPYC-24 -i ${implementation} -c

done

for implementation in AOCL CSR5 Vec-CSR Merge-CSR MKL-IE Naive-CSR SELL-C-s SparseX
do
    clear && python3 main.py -m mlp -s AMD-EPYC-24 -i ${implementation}

done



clear && python3 main.py -m svr -s AMD-EPYC-24 -c -i None

# TODO run svr and tree with cache and without implementation
# TODO run mlp on all