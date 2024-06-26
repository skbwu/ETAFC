# create "outputs" folder if it doesn't exist
if [ ! -d "outputs" ]; then
    mkdir "outputs"
    echo "Created 'outputs' folder."
fi

# create "errors" folder if it doesn't exist
if [ ! -d "errors" ]; then
    mkdir "errors"
    echo "Created 'errors' folder."
fi

# create "results" folder if it doesn't exist
if [ ! -d "results" ]; then
    mkdir "results"
    echo "Created 'results' folder."
fi

# iterate through our settings, indexed by this integer
for ((i = 0; i < 336; i++)); do

    # launch our job
    sbatch dcrnn_main_runscript.sh $i
    sleep 0.5

done

