stime=$(date | awk '{OFS="-"}{print $2, $3, $4}')
nohup python -u algorithms/actor_critic/run.py --start_time=$stime | tee models/actor_critic/$stime.log &
