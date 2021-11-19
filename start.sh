time=$(date | awk '{OFS="-"}{print $2, $3, $4}')
nohup python -u algorithms/actor_critic/run.py --time=$time --comment=$1 > logs/$time.log &
