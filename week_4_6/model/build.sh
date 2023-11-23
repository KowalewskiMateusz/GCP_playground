#/bin/bash
set -e

TAG=$PIPELINE_BASE_IMAGE

docker build --no-cache -t $TAG .

for i in "$@" ; do
    if [[ $i == "--push" ]] ; then
        echo "Pushing image"
        docker push $TAG
    elif [[ $i == "--run" ]] ; then
        echo "Running container"
        docker run $TAG
    fi
done