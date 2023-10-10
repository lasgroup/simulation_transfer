from typing import Any, Dict
import argparse
import pickle
import time
import jax.numpy as jnp

def _load_data(data_load_path: str) -> Any:
    # loads the pkl file
    with open(data_load_path, 'rb') as f:
        data = pickle.load(f)
    return data


def _dump_model(model: Any, model_dump_path: str) -> None:
    # dumps the model in the model_dump_path
    with open(model_dump_path, 'wb') as f:
        pickle.dump(model, f)


def train_model_based_policy(train_data: Dict, sleep_time=5) -> Dict:
    assert 'x_train' in train_data and 'y_train' in train_data
    x_train, y_train = train_data['x_train'], train_data['y_train']

    print(f'sleeping for {sleep_time} sec')
    time.sleep(sleep_time)

    trained_model = {'mean': jnp.mean(x_train, axis=0), 'num_points': x_train.shape[0]}
    return trained_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_load_path', type=str, required=True)
    parser.add_argument('--model_dump_path', type=str, required=True)
    args = parser.parse_args()

    # load data
    function_args = _load_data(args.data_load_path)
    train_data = function_args['train_data']
    kwargs = function_args['kwargs']
    print(f'[Remote] Executing train_model_based_policy function ... ')
    trained_model = train_model_based_policy(train_data, **kwargs)
    _dump_model(trained_model, args.model_dump_path)
    print(f'[Remote] Dumped trained model to {args.model_dump_path}')




