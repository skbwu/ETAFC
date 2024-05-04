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

# create "linear_results" folder if it doesn't exist
if [ ! -d "linear_results" ]; then
    mkdir "linear_results"
    echo "Created 'linear_results' folder."
fi

# iterate through our settings, indexed by this integer
for ((i = 0; i < 144; i++)); do

    # launch our job
    sbatch linear_main_runscript.sh $i
    sleep 0.5

done

