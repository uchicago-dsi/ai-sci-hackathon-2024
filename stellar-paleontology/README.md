# Machine learning enhanced stellar paleontology

## Getting started

Launch an interactive `SLURM` job

```console
sinteractive --account=pi-dfreedman -p schmidt-gpu --gres=gpu:1 --qos=schmidt --time 2:00:00
```

Follow these instructions to launch [`jupyter`](https://rcc-uchicago.github.io/user-guide/software/apps-and-envs/python/?h=jupy#running-jupyterlab).

```console
HOST_IP=`/sbin/ip route get 8.8.8.8 | awk '{print $7;exit}'`
echo $HOST_IP
PORT_NUM=$(shuf -i15001-30000 -n1)
jupyter-lab --no-browser --ip=$HOST_IP --port=$PORT_NUM
```

You should now be able to access the `jupyter` server using a URL from the logs.