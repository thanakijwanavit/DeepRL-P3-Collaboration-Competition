current_date=`date +"%Y-%m-%d-%T"`
echo $current_date
aws2 s3 cp ./results s3://nicrl/results/p3/$current_date/ --recursive
aws2 s3 cp ./models s3://nicrl/results/p3/$current_date/model/ --recursive
