

```
──────────
┏━┳━┳━┓┏━┓
┗┓┃┏┫╋┗┫┻┫
╋┗━┛┗━━┻━┛
──────────
```


# VAE - Tabular Data Generation.

*Variational Auto-encoder for Tabular Data Generation*

1. Capture data distributions
2. Capture a latent low-dim embedding that can be viz using t-SNE




<p>
<img src="https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=darkgreen" />
<img src="https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white" />
<img src="https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white" />
<img src="https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
</p>

## Getting Started



### Prerequisites

| Package     | Version      |
|:----------------|:---------------|
scikit-learn|0.24.2
torch|1.9.0
pandas|1.1.4
numpy|1.18.5
scipy|1.7.1


### Installing

```powershell
pip install -r requirements.txt
```



### Data

Record Linkage Datasets.
<a href="" target="_blank">[Source]</a>


```powershell
/vae-record-generator/datasets/taxi.csv
```


cat\_0|cat\_1|cat\_2|num\_3|num\_4|cat\_5|cat\_6|target
-----|-----|-----|-----|-----|-----|-----|-----
5|20|6,080|-1.0|-1.0|Economy|private|False
5|14|6,080|18.802|25.217|Standard|private|True
6|14|6,080|6.747000000000001|9.8|Economy|private|False
2|6|6,080|-1.0|-1.0|Economy|private|True
3|16|6,080|12.383|19.25|Economy|private|True



## Running 
```python

data, discrete_columns = read_csv(
                        'datasets/taxi.csv',
                        True,
                        '0,1,2,5,6,7'
                        )

print('***')

model = VAE()
model.fit(data, tuple(discrete_columns))

print('***')

sampled = model.sample(10)
print(sampled)
```



or

From */examples/gen_taxi_dataset.py* directory run

```powershell
python gen_taxi_dataset.py
```


#### Output

```powershell
   cat_0  cat_1  cat_2      num_3      num_4    cat_5    cat_6  target
0      5      9   6080   2.089625  22.754222  Economy  private    True
1      5      8   6080  14.775115   7.839013  Economy  private   False
2      5     14   6080   5.143162  36.512501  Economy  private    True
3      5     20   6080  31.609187   5.306750  Economy  private    True
4      5      9   6080   4.031172  25.685939  Economy  private    True
5      5     20   6080  12.516889  12.730164  Economy  private    True
6      5      9   6080   1.528751   7.231834  Economy  private    True
7      5      7   6080  16.058466  23.303466  Economy  private   False
8      5      8   6080  31.659141   8.828095  Economy  private    True
9      6      7   6080   4.533512  27.453236  Economy  private    True
```





## Logs

Parameter values of each run is captured under 
```powershell
\logs
```

## Weights

Pretrained weights in 
```powershell
\weights
```

## Loss

![ELBO](examples/img/elbo.JPG)

## Folder Structure
```powershell
|
|   LICENSE                      
|   README.md                    
|   requirements.txt             
|   run.py
|
+---datasets
|       taxi.csv
|
+---examples
|   |   gen_taxi_dataset.py
|   |
|   \---img
|           elbo.jpg
|
+---models
|   |   base.py
|   |
|   +---vae
|   |   |   Decoder.py
|   |   |   Encoder.py
|   |   |   VAE.py
|
+---utils
|   |   file_utils.py
|   |   transformer.py
|   |   _transformer.py
|
\---weights
        vae.pt
        w1.pt
        w2.pt
```

## License

MIT

## Future


 * Visualizations

## Paper

```
@inproceedings{xu2019modeling,
  title={Modeling Tabular data using Conditional GAN},
  author={Xu, Lei and Skoularidou, Maria and Cuesta-Infante, Alfredo and Veeramachaneni, Kalyan},
  booktitle={Advances in Neural Information Processing Systems},
  year={2019}
}
```