stime = $(date | awk '{OFS="-"}{print $2, $3, $4}')
nohup python algorithms/actor_critic/run.py --start_time=$stime &> models/actor_critic/$stime.log &
