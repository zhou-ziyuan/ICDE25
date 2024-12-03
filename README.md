# ICDE25
These codes are modified version based on repositories from https://github.com/oxwhirl/pymarl and https://github.com/PKU-RL/FOP-DMAC-MACPF/tree/main.

# Run an experiment 
## Train models based on clean observations

```shell
python src/main.py --config=qmix --env-config=sc2 with env_args.map_name=8m Number_attack=0
```

## Train models based on ATSA

```shell
python src/main.py --config=qmix --env-config=sc2 with env_args.map_name=8m Number_attack=8 attack_method=fop_adv_tar 
```

## Testing
```shell
python src/main.py --config=qmix --env-config=sc2 with env_args.map_name=8m evaluate=True Number_attack=8 attack_method=fop_adv_tar checkpoint_path=results/xxx adv_checkpoint_path=results/xxx
```
