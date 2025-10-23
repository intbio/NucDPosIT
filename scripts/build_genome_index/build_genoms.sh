for arg in "$@"; do
    mkdir "$arg/index" 
    bowtie2-build $arg --threads 16
done