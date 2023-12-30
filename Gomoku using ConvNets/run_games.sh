#!/bin/bash

mkdir -p outputs

for i in {100..2000}
do
    python3 compete.py -b 15 -w 5 -x Minimax -o Minimax > "outputs/game${i}.txt"
    echo "Game $i completed"
done

echo "All games completed."
