for model in tree svr mlp
do
    clear && python3 main.py -m ${model} -s AMD-EPYC-24 -i None
    clear && python3 main.py -m ${model} -s AMD-EPYC-24 -i None -c
done
# TODO run svr and tree with cache and without implementation
# TODO run mlp on all