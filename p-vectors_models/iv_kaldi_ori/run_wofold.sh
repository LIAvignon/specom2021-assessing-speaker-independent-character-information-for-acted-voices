set -e

# -- DEFINE VARIABLES AND CREATE DIRECTORIES
MOTHER_DIR="../data/masseffect/prot_zero_wotest"
EXP_DIR="exp/prot_zero_wofold/iv"
LST_DIR="${EXP_DIR}/lst"
FEATURES_FILE="../vocal_similarity_system/data/masseffect_ivectors.txt"
TEMPERATURE="1.0"


mkdir -p ${EXP_DIR}
mkdir ${LST_DIR}
cp -r ${MOTHER_DIR}/* ${EXP_DIR}


# -- COPY SCRIPT
cp $0 ${EXP_DIR}



# -- CHANGE LIST IF MISLEADER : "from_to.pkl" file
if [ -f "${LST_DIR}/from_to.pkl" ]
then
        lst_changes="${LST_DIR}/from_to.pkl"
        new_features_file="${EXP_DIR}/features_modified.txt"
        echo "[switch] Begin "
        python bin/change_locutors.py $lst_changes "$LST_DIR/train_fr.lst" --outfile "$LST_DIR/train_fr.lst"
        python bin/change_locutors.py $lst_changes "$LST_DIR/train.lst" --outfile "$LST_DIR/train.lst"

        python bin/change_locutors.py $lst_changes "$LST_DIR/val_fr.lst" --outfile "$LST_DIR/val_fr.lst"
        python bin/change_locutors.py $lst_changes "$LST_DIR/val.lst" --outfile "$LST_DIR/val.lst"

        python bin/change_locutors.py $lst_changes "${FEATURES_FILE}" --outfile "${new_features_file}"
        FEATURES_FILE=${new_features_file}
        echo "[switch] End"
fi





# -- TRAIN TEACHER
teachermdldir="${EXP_DIR}/${TEMPERATURE}/teacher"
mkdir -p ${teachermdldir}
python bin/train-teacher.py "${LST_DIR}/train.lst" "${LST_DIR}/val.lst" ${FEATURES_FILE} ${teachermdldir} --batch 12 --temperature ${TEMPERATURE} --epochs 100 --drop-rate 0.35 --emb-dim 64


