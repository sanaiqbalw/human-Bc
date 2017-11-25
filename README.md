# human-Bc


ensemble human pref with behavioural cloning

### python learn.py --expert_policy_file experts/Hopper-v1.pkl --envname Hopper-v1 --num_rollouts 3 --bc_training_epochs 3 --pretrain_labels 2 --num_iters 3



1. file used right now for reward function is actual_reward_file.py

2. file used for policy function policy.py

3. Commented segmentor in comparison.py, because its not showing videos to label. Also all labels are are stored as NULL. I have instead saved a comaprison0.json with hard coded labels to test run the file.

4. Path in collector.py changed
