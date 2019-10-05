#!/bin/bash
#SBATCH -N 1
#SBATCH -t 5:00:00
#SBATCH --mem=70G

module load pre2019
module load Python/3.6.1-intel-2016b 

echo "Stat job $PBS_JOBID started at `date`"


#mkdir "$TMPDIR"/ThesisII
#mkdir "$TMPDIR"/ThesisII/data
#cp -r $HOME/ThesisII/data/"$lang" "$TMPDIR"/ThesisII/data
#cp -r $HOME/ThesisII/filtering "$TMPDIR"/ThesisII
#cp -r $HOME/ThesisII/stats "$TMPDIR"/ThesisII


rsync -a $HOME/ThesisII "$TMPDIR"/ --exclude data --exclude .git
cd "$TMPDIR"/ThesisII

echo

lang=FI

mkdir data
cp -r $HOME/ThesisII/data/"$lang"_pkl "$TMPDIR"/ThesisII/data/
cp -r $HOME/ThesisII/data/reader.py "$TMPDIR"/ThesisII/data/
cp -r $HOME/ThesisII/data/corpus.py "$TMPDIR"/ThesisII/data/


# 2 4 8 16 32
for f in 4; do

python3.6 TF_main_parallelised.py --lang=$lang --n_tokens=2500000 --factor=$f

echo 
echo "done with factor $f at `date`"
echo 

done


cp -r $TMPDIR/ThesisII/results/"$lang"/ $HOME/ThesisII/results


echo "Job $PBS_JOBID ended at `date`"