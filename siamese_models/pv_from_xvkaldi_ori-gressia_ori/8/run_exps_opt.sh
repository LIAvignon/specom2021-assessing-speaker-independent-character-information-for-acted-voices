

if [ "$1" == "help" ]
then
	echo "Description:"
	echo "This program runs a script n times. The current loop n value is stored in exp_num variable that you can use in your bash script. (using \$exp_num)"
	echo
	echo "To execute this script:"
	echo "${0} [SCRIPT] [MIN] [MAX] [DEVICE]"
	echo
	echo "Arguments details:"
	echo "Script: script to execute"
	echo "Min: minimum value of n"
	echo "Max: maximum value of n"
	echo "Device: device variable value for neural net training."
	echo
	echo "To show helper message: "
	echo "${0} help"

	exit 1
fi

if [[ "$#" -ne 4 ]]
then
	echo "This script needs four arguments but received $#. For more details:"
	echo "${0} help"
	exit 1
fi

SCRIPT=$1
MIN=$2
MAX=$3
DEVICE=$4

shift 4 # Shift to pass arguments to the scripts

for exp_num in `seq ${MIN} ${MAX}`
do
	source $SCRIPT
	cp $SCRIPT ${EXP_DIR}
done

