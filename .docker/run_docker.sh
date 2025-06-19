# Precisa desse comando todas as vezes que for rodar o docker
xhost si:localuser:root

dcoker run --gpus all -it --rm \
           -v $(pwd):/workspace \
           --entrypoint /workspace/entrypoint.sh \
           --privileged \
           --name arniqa \
           --net=host \
           -v /tmp/.X11-unix:/tmp/.X11-unix \
           --volume="$HOME/.Xauthority:/root/.Xauthority:rw" \
           --rm \
           bash
