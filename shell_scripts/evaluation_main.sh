#!/bin/bash
#SBATCH -N 1
#SBATCH -t 30:00:00
#SBATCH --mem=70G

module load pre2019
module load Python/3.6.1-intel-2016b 

echo "evaluation_main job $PBS_JOBID started at `date`"

rsync -a $HOME/ThesisII "$TMPDIR"/ --exclude data --exclude .git

cd "$TMPDIR"/ThesisII

mkdir data
cp $HOME/ThesisII/data/reader.py "$TMPDIR"/ThesisII/data/
cp $HOME/ThesisII/data/corpus.py "$TMPDIR"/ThesisII/data/

lang=FI

echo "language: $lang"

cp -r $HOME/ThesisII/data/"$lang"_pkl "$TMPDIR"/ThesisII/data/

python3 typicality_eval.py --lang=$lang --factors 2 18 22 --hist_lens 2 64 81 
    
echo
echo "done with language $lang at `date`"

cp -r $TMPDIR/ThesisII/results/$lang/evaluation $HOME/ThesisII/results/$lang/

echo "and copied"
echo


echo "Job $PBS_JOBID ended at `date`"