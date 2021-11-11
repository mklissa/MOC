# for d in ppo2/checkpoints/* ; do
# 	rm $d/*.txt;rm $d/*.csv
# 	# folder="${d}/checkpoints/*"
#  #    mv $folder $d
#  #    rm -r "${d}/checkpoints"
# done


for d in ppo2/results/* ; do
	rm -r $d/checkpoints
	# folder="${d}/checkpoints/*"
 #    mv $folder $d
 #    rm -r "${d}/checkpoints"
done
