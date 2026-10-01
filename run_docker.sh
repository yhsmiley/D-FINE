WORKSPACE=/media/data/D-FINE

docker run -it --rm \
    --gpus all \
    -w $WORKSPACE \
    -v $WORKSPACE:$WORKSPACE \
    dfine
