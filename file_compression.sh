scp ~/scratch/Outlier/cls/clkk_000999.npy dl:~/www/columbialensing/outlier
scp ~/scratch/Outlier/GRFs/GRF_000999.fits dl:~/www/columbialensing/outlier

tar -cf ~/scratch/Outlier/GRFs.tar ~/scratch/Outlier/GRFs
scp ~/scratch/Outlier/GRFs.tar dl:~/www/columbialensing/outlier/
scp ~/work/Outlier/GRF_params_output.txt dl:~/www/columbialensing/outlier/
