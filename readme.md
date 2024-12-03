## Install
```bash 
conda create -n mvgaussian python=3.8
conda activate mvgaussian
pip install -r requirements.txt
conda install -c fvcore -c iopath -c conda-forge fvcore
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu117_pyt200/download.html
```

## Data
```bash
- data
    - ours_data
        - napoleon
            - napoleon.obj
```

## Run
```bash
python run_experiment.py --config config/sdxl_config.yaml
```

then colmap data dir will be saved in save_dir

put the colmap dir for gaussian splatting

**3d gaussian**
```bash
python train.py -s data/colmap
```