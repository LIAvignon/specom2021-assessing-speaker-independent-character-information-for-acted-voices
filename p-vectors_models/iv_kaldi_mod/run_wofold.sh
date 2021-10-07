set -e

# -- DEFINE VARIABLES AND CREATE DIRECTORIES
MOTHER_DIR="../data/masseffect/prot_zero_wotest_switch"
EXP_DIR="exp/prot_zero_wofold/iv_switch"
LST_DIR="${EXP_DIR}/lst"
FEATURES_FILE="../vocal_similarity_system/data/masseffect_ivectors.txt"
TEMPERATURE="1.0"

lst_masseffect="../vocal_similarity_system/data/metas.lst"



mkdir -p ${EXP_DIR}
mkdir ${LST_DIR}
cp -r ${MOTHER_DIR}/* ${EXP_DIR}


# -- COPY SCRIPT
cp $0 ${EXP_DIR}



# -- CHANGE LIST IF MISLEADER : "from_to.pkl" file
ORIGINAL_FEATURES_FILE="${FEATURES_FILE}"
original_lst_masseffect="${lst_masseffect}"

if [ -f "${LST_DIR}/from_to.pkl" ]
then
        lst_changes="${LST_DIR}/from_to.pkl"
        new_features_file="${EXP_DIR}/masseffect_features_modified.txt"
        new_lst_masseffect="${EXP_DIR}/lst_modified.lst"
        echo "[switch] Begin "
        python bin/change_locutors.py $lst_changes "$LST_DIR/train_fr.lst" --outfile "$LST_DIR/train_switch_fr.lst"
        python bin/change_locutors.py $lst_changes "$LST_DIR/train.lst" --outfile "$LST_DIR/train_switch.lst"

        python bin/change_locutors.py $lst_changes "$LST_DIR/val_fr.lst" --outfile "$LST_DIR/val_switch_fr.lst"
        python bin/change_locutors.py $lst_changes "$LST_DIR/val.lst" --outfile "$LST_DIR/val_switch.lst"

        python bin/change_locutors.py $lst_changes "${FEATURES_FILE}" --outfile "${new_features_file}"
	python bin/change_locutors.py $lst_changes "${original_lst_masseffect}" --outfile "${new_lst_masseffect}"
        FEATURES_FILE=${new_features_file}
        lst_masseffect=${new_lst_masseffect}
        echo "[switch] End"
fi





# -- TRAIN TEACHER
teachermdldir="${EXP_DIR}/${TEMPERATURE}/teacher"
mkdir -p ${teachermdldir}
python bin/train-teacher.py "${LST_DIR}/train_switch.lst" "${LST_DIR}/val_switch.lst" ${FEATURES_FILE} ${teachermdldir} --batch 12 --temperature ${TEMPERATURE} --epochs 100 --drop-rate 0.35 --emb-dim 64

python bin/embedding-extract.py $lst_masseffect ${FEATURES_FILE} $teachermdldir --temperature ${TEMPERATURE}

python bin/evaluate-model.py "${teachermdldir}/model_checkpoint.h5" "${FEATURES_FILE}" "${LST_DIR}/val_switch.lst" "${teachermdldir}/label_encoder.pkl" > "${teachermdldir}/eval_val.log"

#python bin/vector-clustering.py "${LST_DIR}/test.lst" "${teachermdldir}/masseffect_pvectors.txt" ${teachermdldir} --prefix "test_" > "${teachermdldir}/test_clustering.log"

#python bin/vector-clustering.py "${LST_DIR}/val.lst" "${teachermdldir}/masseffect_pvectors.txt" ${teachermdldir} --prefix "val_" > "${teachermdldir}/val_clustering.log"
