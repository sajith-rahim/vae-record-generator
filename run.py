

from models.vae.VAE import VAE
from utils.file_utils import read_csv


def run():
    data, discrete_columns = read_csv('datasets/taxi.csv', True, '0,1,2,5,6,7')
    # data = data.iloc[1:500, :]
    print('***')

    model = VAE()
    model.fit(data, tuple(discrete_columns))

    sampled = model.sample(10)
    print(sampled)


if __name__ == '__main__':
    run()
