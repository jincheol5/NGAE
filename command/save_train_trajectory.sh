#!/bin/bash
set -e

for graph_type in ladder grid tree erdos_renyi barabasi_albert community caveman
do
    python -m app.save_trajectory --mode train --n_node 20 --graph_type $graph_type
done
