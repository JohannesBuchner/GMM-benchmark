export PYTHONPATH=.:/mnt/data/daten/PostDoc2/programming/GGMM/:$HOME/Downloads/gmmx/

for f in 0.5 0.95
do
    for i in {eFEDS,COSMOS}_z[0-9]*_pred_mags.txt.gz
    do 
        python3 -O test_lightgmm_approximations.py $i $f
    done
done

for i in 5000 50000
do
    python3 pareto_analysis.py {eFEDS,COSMOS}_z*_lightgmm${i}.data
    mv pareto_analysis.pdf pareto_analysis_lightgmm${i}.pdf
done

