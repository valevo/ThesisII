#!/bin/bash
#SBATCH -N 1
#SBATCH -t 12:00:00
#SBATCH --mem=25G

module load pre2019
module load Python/3.6.1-intel-2016b 

pip install --user lexical-diversity
pip install --user tag-fixer
python3 -c "from lexical_diversity import lex_div; print(lex_div)"

echo "evaluation_main job $PBS_JOBID started at `date`"

rsync -a $HOME/ThesisII "$TMPDIR"/ --exclude data --exclude .git

cd "$TMPDIR"/ThesisII

#mkdir data
#cp $HOME/ThesisII/data/reader.py "$TMPDIR"/ThesisII/data/
#cp $HOME/ThesisII/data/corpus.py "$TMPDIR"/ThesisII/data/

argstogether=("--lang EO --factors 2 6 10 14 18 22 26 --hist_lens 2 4 8 16 32"
              "--lang FI --factors 2 6 10 14 18 22 --hist_lens 2 4 8 16 32 64 81"
              "--lang ID --factors 2 6 10 14 18 --hist_lens 2 4 8 16 32 64 81"
              "--lang KO --factors 2 6 10 14 --hist_lens 2 4 8 16 32 64 81"
              "--lang NO --factors 2 6 10 14 18 22 26 --hist_lens 2 4 8 16 32"
              "--lang TR --factors 2 --hist_lens 2 4 8 16 32 64 81"
              "--lang VI --factors 2 6 10 14 18 --hist_lens 2 4 8 16 32")

cp -r $HOME/ThesisII/data/ "$TMPDIR"/ThesisII/data/   
              
for a in "${argstogether[@]}"; do

    echo "args: $a"

    python3 typicality_eval.py $a
    echo
    echo "done with typicality evaluation at `date`"
    

    python3 diversity_eval.py $a
    echo
    echo "done with diversity evaluation at `date`"

    python3 normality_eval.py $a

    echo
    echo "done with normality evaluation at `date`"
done


echo
echo
echo
echo "Copying results..."


langs="EO FI ID KO NO TR VI"
for lang in "${langs[@]}"; do
    cp -vr $TMPDIR/ThesisII/results/$lang/evaluation $HOME/ThesisII/results/$lang/
    echo "Copied $lang"
done

echo "Job $PBS_JOBID ended at `date`"
    
