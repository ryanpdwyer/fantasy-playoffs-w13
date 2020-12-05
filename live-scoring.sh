#! /bin/bash

function updateScoring {

    echo `date +"%A %I:%M %p"` > update_time.txt
    python sleeper_sel.py
    git commit -am "Auto-update: `date`"
    git push

}


gsed -i 's/auto_update = False/auto_update = True/g' first_app.py

while :
do 
    echo "Press Ctrl+C to stop"
    updateScoring
    sleep 300 # Wait 5 minutes
done

gsed -i 's/auto_update = True/auto_update = False/g' first_app.py