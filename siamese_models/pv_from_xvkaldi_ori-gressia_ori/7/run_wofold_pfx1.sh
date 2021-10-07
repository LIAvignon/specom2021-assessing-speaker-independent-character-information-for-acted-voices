set -e


# CONFIGURATION VARIABLES
MOTHER_DIR="/local_disk/pegasus/laboinfo/mquillot/data/masseffect/prot_zero_wotest"
EXP_DIR="exp/prot_zero_wofold/pfx_10times_rw/1"
FEATURES_FILE="/local_disk/pegasus/laboinfo/mquillot/knowledge_distillation/exp/prot_zero_wofold/xv/1.0/teacher/masseffect_pvectors.txt"
INPUT_DIM=64

#TRAIN_NUM_MIN=1
#TRAIN_NUM_MAX=10

# INIT_WEIGHTS="exp/prot_zero_wofold/xv_switch_fullr_10times/1/1/model/init_weights.h5"
# END CONFIGURATION VARIABLES





# Begin the program
if [ -z "$DEVICE" ]
then
	DEVICE=1
fi

export CUDA_VISIBLE_DEVICES=${DEVICE}
if [ ! -z "$exp_num" ]
then
	EXP_DIR="${EXP_DIR}/${exp_num}"
fi

echo "EXP_DIR: " ${EXP_DIR}

LST_DIR="${EXP_DIR}/lst"
ARRAY_DIR="${EXP_DIR}/array"
MDL_DIR="${EXP_DIR}/model"



mkdir -p ${EXP_DIR}
mkdir ${LST_DIR}
mkdir ${ARRAY_DIR}
mkdir ${MDL_DIR}
cp -r ${MOTHER_DIR}/* ${EXP_DIR}



# COPY SCRIPT INTO EXP
cp $0 ${EXP_DIR}



# CHANGE LIST IF MISLEADER : "from_to.pkl" file
if [ -f "$LST_DIR/from_to.pkl" ]
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
	echo
fi

# MAKE TRIALS
echo "[make-trials] Begin"
python bin/make-trials.py "${LST_DIR}/train_en.lst" "${LST_DIR}/train_fr.lst" ${FEATURES_FILE} ${ARRAY_DIR}/train --meta-data "data/meta.csv" --separator "," --balance --verbose > "${LST_DIR}/train_trials.lst"
python bin/make-trials.py "${LST_DIR}/val_en.lst" "${LST_DIR}/val_fr.lst" ${FEATURES_FILE} ${ARRAY_DIR}/val --meta-data "data/meta.csv" --separator "," --balance --verbose > "${LST_DIR}/val_trials.lst"
echo "[make-trials] End"
echo


# REPLACE TEST FEATURES
echo "[replace test features] Begin"
python3 bin/export-features.py "${EXP_DIR}/lst/test_fr.lst" "${FEATURES_FILE}" --output "${EXP_DIR}/array/test/french_feats.npy"
python3 bin/export-features.py "${EXP_DIR}/lst/test_en.lst" "${FEATURES_FILE}" --output "${EXP_DIR}/array/test/english_feats.npy"
echo "[replace test features] End"
echo

# IF ALREADY EXISTS TEST_SWITCH
if [ -d "$ARRAY_DIR/test_switch" ]
then
	# test_switch is pre-computed in order to share the exact same test for all the systems.
	echo "[replace test switch features] Begin"
	python3 bin/export-features.py "${EXP_DIR}/lst/test_fr_switch.lst" "${FEATURES_FILE}" --output "${EXP_DIR}/array/test_switch/french_feats.npy"
	python3 bin/export-features.py "${EXP_DIR}/lst/test_en_switch.lst" "${FEATURES_FILE}" --output "${EXP_DIR}/array/test_switch/english_feats.npy"
	echo "[replace test_switch features] End"
	echo
fi



# MANAGE TRAIN NUMBER VARIABLES
if [ -z "${TRAIN_NUM_MIN}" ]
then
	train_num_min=1
	train_num_max=1
else
	train_num_min=${TRAIN_NUM_MIN}
	train_num_max=${TRAIN_NUM_MAX}
fi


# LAUNCH EXPERIMENT
for num_exp in `seq ${train_num_min} ${train_num_max}`
do
        init_weights_option=""
        if [ ! "$INIT_WEIGHTS" = "" ]
        then
		init_weights_option="--init-weights ${INIT_WEIGHTS}"
	fi
	
	sub_mdl_dir="${EXP_DIR}/model"
        if [ ! -z "$TRAIN_NUM_MIN" ]
	then
        	sub_mdl_dir="${EXP_DIR}/exp_${num_exp}"
		mkdir ${sub_mdl_dir}
	fi
	echo "[OPTIONS AND VARIABLES] Begin"
	echo "- INIT WEIGHTS OPTION: ${init_weights_option}"
	echo "- SUB MODEL DIR : ${sub_mdl_dir}"
	echo "[OPTIONS AND VARIABLES] End"
	echo

	echo "[TRAINING AND SCORING] Begin"
	set -x
	python bin/train-siamese.py "$ARRAY_DIR/train" "$ARRAY_DIR/val" $sub_mdl_dir --input-dim ${INPUT_DIM} --dropout 0.3 --num-epochs 100 --batch-size 64 ${init_weights_option} > "${EXP_DIR}/train_siamese.log" 2> "${EXP_DIR}/train_siamese_error.log"

	python bin/score-siamese.py "$ARRAY_DIR/val" $sub_mdl_dir --input-dim ${INPUT_DIM} > "${EXP_DIR}/score_siamese_val.log" 2> "${EXP_DIR}/score_siamese_val_error.log"

	python bin/score-siamese.py "$ARRAY_DIR/test" $sub_mdl_dir --input-dim ${INPUT_DIM} > "${EXP_DIR}/score_siamese_test.log" 2> "${EXP_DIR}/score_siamese_test_error.log"
	
	if [ -d "$ARRAY_DIR/test_switch" ]
	then
		python bin/score-siamese.py "$ARRAY_DIR/test_switch" $sub_mdl_dir --input-dim ${INPUT_DIM} > "${EXP_DIR}/score_siamese_switch_test.log" 2> "${EXP_DIR}/score_siamese_switch_test_error.log"
	fi
	set +x
	echo "[TRAINING AND SCORING] End"
	echo
done
