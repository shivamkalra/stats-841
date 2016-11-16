#!/bin/bash
#PBS -S /bin/bash
#PBS -N TASK
#PBS -l nodes=1:ppn=6
#PBS -l walltime=04:00:00
#PBS -q mpi
# Add any addition to your environment variables like PATH. For example, if your
# local python installation is in $HOME/python
module unload gcc python intel openmpi
module load gcc

export LD_LIBRARY_PATH=${PYPATH}/lib:${LD_LIBRARY_PATH}
export -n LDFLAGS

PROJECT_DIR="/work/s6kalra/projects/stats-841-project-shivam"
PYTHON_EX="${HOME}/stats-841-env/bin/python"

# KERAS_BACKEND=theano THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 $PYTHON_EX $PROJECT_DIR/test_keras.py
ENVIRONMENT=sharcnet MPLBACKEND="agg" $PYTHON_EX -m scoop $PROJECT_DIR/source/create_spectrogram.py
