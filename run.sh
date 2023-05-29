
list_method=('Energy' 'RW_Energy')
list_dataset=('iNaturalist' 'SUN' 'Places' 'Textures' )

for method in ${list_method[*]}
do
for dataset in ${list_dataset[*]}
do
bash ./test.sh $method $dataset ./checkpoints/resnet101.pth \
    result_log/$method ./meta/train_LT_a8.txt 8
done
done

