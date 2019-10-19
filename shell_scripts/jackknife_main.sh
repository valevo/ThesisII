#!/bin/bash
#SBATCH -N 1
#SBATCH -t 5:00:00
#SBATCH --mem=70G

module load pre2019
module load Python/3.6.1-intel-2016b 

echo "data_main job $PBS_JOBID started at `date`"

rsync -a $HOME/ThesisII "$TMPDIR"/ --exclude data --exclude .git

mkdir data
cp -r $HOME/ThesisII/data/reader.py "$TMPDIR"/ThesisII/data/
cp -r $HOME/ThesisII/data/corpus.py "$TMPDIR"/ThesisII/data/

cd "$TMPDIR"/ThesisII

for lang in EO FI ID KO NO TR VI; do
    echo "language: $lang"
    cp -r $HOME/ThesisII/data/"$lang"_pkl "$TMPDIR"/ThesisII/data/
    
    python3 jackknife_main.py --lang=$lang
    
    echo
    echo "done with language $lang at `date`"

    cp -r $TMPDIR/ThesisII/results/$lang/jackknife $HOME/ThesisII/results/$lang/

    echo "and copied"
    echo
done

# cp -r $TMPDIR/ThesisII/results/"$lang"/ $HOME/ThesisII/results


echo "Job $PBS_JOBID ended at `date`"