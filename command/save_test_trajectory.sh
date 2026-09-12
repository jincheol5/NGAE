#!/bin/bash
set -e

for n_node in 20 50 100
do
    for graph_type in ladder grid tree erdos_renyi barabasi_albert community caveman
    do
        python -m app.save_trajectory --mode test --n_node $n_node --graph_type $graph_type
    done
done