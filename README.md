# PlanningWithWorldModels

## Checkpoints

Model checkpoints are stored with [Git LFS](https://git-lfs.com). After cloning:

```sh
git lfs pull
./reassemble.sh
```

`tetris/ckpt/*/agent.pkl` is larger than GitHub's 2 GiB LFS file limit, so it is
stored as `agent.pkl.part-*` chunks; `reassemble.sh` joins them and verifies the
checksum.

The `tetris_400m_latest` checkpoint (`agent.pkl`, 5.2 GB) and its replay buffer
are not included; only that run's config and logs are.
