#!/bin/bash
#SBATCH -N 1
#SBATCH -t 5:00:00
#SBATCH --mem=30G


module load Python/3.6.1-intel-2016b 

echo "Stat job $PBS_JOBID started at `date`"


lang=FI

#mkdir "$TMPDIR"/ThesisII
#mkdir "$TMPDIR"/ThesisII/data
#cp -r $HOME/ThesisII/data/"$lang" "$TMPDIR"/ThesisII/data
#cp -r $HOME/ThesisII/filtering "$TMPDIR"/ThesisII
#cp -r $HOME/ThesisII/stats "$TMPDIR"/ThesisII


rsync -a --progress $HOME/ThesisII "$TMPDIR"/ --exclude data
cd "$TMPDIR"/ThesisII
mkdir data
cp -r $HOME/ThesisII/data/"$lang" "$TMPDIR"/ThesisII/data/
cp -r $HOME/ThesisII/data/reader.py "$TMPDIR"/ThesisII/data/
cp -r $HOME/ThesisII/data/corpus.py "$TMPDIR"/ThesisII/data/


python3.6 SRF_main.py --lang=$lang --n_tokens=1000000 --hist_len=2


cp -r $TMPDIR/ThesisII/results/"$lang"/ $HOME/ThesisII/results


echo "Job $PBS_JOBID ended at `date`"