set key autotitle columnhead
N = system("awk 'NR==1{print NF}' AvgStressTensor.txt")
plot for [i=2:N] "AvgStressTensor.txt" u 1:i w l
set key autotitle columnhead
set term png
set output "avgStress.png"
replot

