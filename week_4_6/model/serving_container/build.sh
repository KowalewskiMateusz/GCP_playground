TAG=$PIPELINE_SERVING_IMAGE

set -e

docker build --no-cache -t $TAG --build-arg PIPELINE_BASE_IMAGE=$PIPELINE_BASE_IMAGE .

for i in "$@" ; do
    if [[ $i == "--push" ]] ; then
        echo "Pushing image"
        docker push $TAG
    elif [[ $i == "--run" ]] ; then
        echo "Running container"
        docker run $TAG
    fi
done