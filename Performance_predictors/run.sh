for model in tree svr mlp
do
    clear && python3 main.py -m ${model} -s AMD-EPYC-24 -i None
    clear && python3 main.py -m ${model} -s AMD-EPYC-24 -i None -c
    for implementation in AOCL CSR5 Vec-CSR Merge-CSR MKL-IE Naive-CSR SELL-C-s SparseX
    do
        clear && python3 main.py -m ${model} -s AMD-EPYC-24 -i ${implementation} -c
        clear && python3 main.py -m ${model} -s AMD-EPYC-24 -i ${implementation}
    done
done

for model in tree svr mlp
do
    clear && python3 main.py -m ${model} -s Tesla-A100 -i None
    clear && python3 main.py -m ${model} -s Tesla-A100 -i None -c
    for implementation in Merge cu-COO cu-CSR
    do
        clear && python3 main.py -m ${model} -s Tesla-A100 -i ${implementation} -c
        clear && python3 main.py -m ${model} -s Tesla-A100 -i ${implementation}
    done
done
