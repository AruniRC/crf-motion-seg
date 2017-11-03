# Run this on Swarm2 (the source of the files)



make_list () {
    DATA_PATH=$1
    OUT_PATH=$2
    FLOW_PATH=$3
    for folder in `ls ${DATA_PATH}`; do
        echo $folder

        # list the images with full path under the folder
        ls -1 ${DATA_PATH}${folder} | grep 'jpg' | awk -v prefix="${DATA_PATH}${folder}/" '{print prefix $0}' \
        > ${OUT_PATH}${folder}'.txt'

        # # first-second frame listings in separate files
        head -n -1 ${OUT_PATH}${folder}'.txt' > ${OUT_PATH}${folder}'-first.txt'
        tail -n +2 ${OUT_PATH}${folder}'.txt' > ${OUT_PATH}${folder}'-second.txt'

        # # list of output .flow filenames (replace .jpg with .flo in ...-first.txt)
        mkdir -p $FLOW_PATH$folder
        ls -1 ${DATA_PATH}${folder} | grep 'jpg' | awk -v prefix="$FLOW_PATH$folder/" '{print prefix $0}' | \
        sed -e 's/.jpg/.flo/g' | head -n -1 > ${OUT_PATH}${folder}'-flow-outputs.txt'
    done
}


# For DAVIS dataset
DATA_PATH=data/DAVIS/DAVIS/JPEGImages/480p/
OUT_PATH=data/DAVIS/input-lists/
FLOW_PATH=data/DAVIS/output-flow/
mkdir -p ${OUT_PATH}
mkdir -p ${FLOW_PATH}
make_list ${DATA_PATH} ${OUT_PATH} ${FLOW_PATH}
