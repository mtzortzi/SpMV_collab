for implementation in AOCL CSR5 Vec-CSR Merge-CSR MKL-IE Naive-CSR SELL-C-s SparseX
do
    clear && python3 main.py -m tree -s AMD-EPYC-24 -i ${implementation}
done
