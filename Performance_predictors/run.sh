for model in tree svr
do
    clear && python3 main.py -m ${model} -s Tesla-A100 -i None
    for implementation in Merge cu-COO cu-CSR
    do
        clear && python3 main.py -m ${model} -s Tesla-A100 -i ${implementation}
    done
done
