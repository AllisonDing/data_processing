# data_processing
structured and unstructured data processing use case

# execution
Step 1: Install Conda Environment
```python
conda create -n rapids-25.12-python-3.13 -c rapidsai -c conda-forge \
    cudf=25.12 cuml=25.12 python=3.13 'cuda-version>=13.0,<=13.1' \
    dash jupyterlab
```

Step 2: Execute on the Terminal
```python
./run.sh
```


