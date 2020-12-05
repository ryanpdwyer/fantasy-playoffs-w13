#! /bin/bash

function updateScoring {

    python sleeper_sel.py
    git commit -am "Auto-update: `date`"
    git push

}

while :
do 
    echo "Press Ctrl+C to stop"
    updateScoring
    sleep 300 # Wait 5 minutes
done