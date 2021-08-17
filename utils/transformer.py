from collections import namedtuple

import numpy as np
import pandas as pd
from sklearn.mixture import BayesianGaussianMixture

from rdt.transformers import OneHotEncodingTransformer

AttrMetaDetails = namedtuple("AttrMetaDetails", ["dim", "activation_fn"])

AttrTransformDetails = namedtuple(
    "AttrTransformDetails", ["column_name", "column_type",
                             "transformer", "transform_aux",
                             "output_info", "output_dimensions"])


class Transformer:
    """
    Each column is transformed independently
    Discrete Columns: 1 hot encoded
    Continuous Columns: Bayesian Gaussian Mixture
                        each value is represented as one hot vector
                        indicating the mode (Beta_i,j) and a scalar value (alpha_i,j)
                        indicating value within mode.
    """

    def __init__(self, n_modes=10, weightage=0.005):
        """Create a data transformer.
        Args:
            n (int):
                Maximum number of Gaussian distributions in Bayesian GMM.
            weightage (float):
                Weight of distribution for a Gaussian distribution to be kept.
        """
        self._n_modes = n_modes
        self._weightage = weightage
        self._attr_dtypes = []
        self._attr_transform_info = []

    def fit_discrete(self, column_name, data):
        """ one hot encoder for discrete column."""

        # ohe = OneHotEncoder()
        # ohe.fit(np.array(raw_column_data).reshape(-1,1))
        # num_categories = len(ohe.get_feature_names())

        ohe = OneHotEncodingTransformer()
        ohe.fit(data)
        num_categories = len(ohe.dummies)

        return AttrTransformDetails(
            column_name=column_name, column_type="discrete", transformer=ohe,
            transform_aux=None,
            output_info=[AttrMetaDetails(num_categories, 'softmax')],
            output_dimensions=num_categories)

    def transform_discrete(self, attr_transform_details, raw_column_data):
        ohe = attr_transform_details.transformer
        d = [ohe.transform(raw_column_data)]
        return d

    def fit_continuous(self, column_name, data):
        """ bayesian GMM for continuous column"""
        gm = BayesianGaussianMixture(
            self._n_modes,
            weight_concentration_prior_type='dirichlet_process',
            weight_concentration_prior=0.001,
            n_init=1
        )

        gm.fit(data.reshape(-1, 1))
        _weights = gm.weights_
        # valid gaussians
        valid_component_indicator = _weights > self._weightage
        num_components = valid_component_indicator.sum()

        return AttrTransformDetails(
            column_name=column_name, column_type="continuous", transformer=gm,
            transform_aux=valid_component_indicator,
            output_info=[AttrMetaDetails(1, 'tanh'), AttrMetaDetails(num_components, 'softmax')],
            output_dimensions=1 + num_components)

    def transform_continuous(self, attr_transform_details, data):
        gmm = attr_transform_details.transformer

        valid_component_indicator = attr_transform_details.transform_aux
        n_components = valid_component_indicator.sum()

        # N(0,1)
        means = gmm.means_.reshape((1, self._n_modes))
        stds = np.sqrt(gmm.covariances_).reshape((1, self._n_modes))

        # alpha_i_j = c_i_j - eta / 4 * phi for_all-> i and valid gaussians
        normalized_values = ((data - means) / (4 * stds))[:, valid_component_indicator]

        # p_k = pi_k * N_k
        component_probs = gmm.predict_proba(data)[:, valid_component_indicator]

        # beta
        selected_component = np.zeros(len(data), dtype='int')

        for i in range(len(data)):
            component_prob_t = component_probs[i] + 1e-6
            component_prob_t = component_prob_t / component_prob_t.sum()
            selected_component[i] = np.random.choice(
                np.arange(n_components), p=component_prob_t)

        selected_normalized_value = normalized_values[
            np.arange(len(data)), selected_component].reshape([-1, 1])

        selected_normalized_value = np.clip(selected_normalized_value, -.99, .99)

        sel_comp_one_hot_encoded = np.zeros_like(component_probs)

        sel_comp_one_hot_encoded[np.arange(len(data)), selected_component] = 1

        return [selected_normalized_value, sel_comp_one_hot_encoded]

    def fit(self, raw_data, discrete_columns=tuple()):
        """
        fit GMM for continuous columns and One hot encoder for discrete columns.
        """
        self.output_info_list = []
        self.output_dimensions = 0

        if not isinstance(raw_data, pd.DataFrame):
            self.is_dataframe = False
            raw_data = pd.DataFrame(raw_data)
        else:
            self.is_dataframe = True

        self._attr_dtypes = raw_data.infer_objects().dtypes

        self._attr_transform_info = []

        discrete_columns = [raw_data.columns[int(i)] for i in discrete_columns]
        for column_name in raw_data.columns:
            raw_column_data = raw_data[column_name].values
            if column_name in discrete_columns:
                _transform_info = self.fit_discrete(
                    column_name, raw_column_data)
            else:
                _transform_info = self.fit_continuous(
                    column_name, raw_column_data)
            # _transform_info : dict
            self.output_info_list.append(_transform_info.output_info)
            self.output_dimensions += _transform_info.output_dimensions
            self._attr_transform_info.append(_transform_info)

    def transform(self, raw_data):
        """"""
        if not isinstance(raw_data, pd.DataFrame):
            raw_data = pd.DataFrame(raw_data)

        column_data_list = []
        for attr_transform_info in self._attr_transform_info:
            column_data = raw_data[[attr_transform_info.column_name]].values
            if attr_transform_info.column_type == "continuous":
                column_data_list += self.transform_continuous(attr_transform_info, column_data)
            else:
                assert attr_transform_info.column_type == "discrete"
                column_data_list += self.transform_discrete(
                    attr_transform_info, column_data)

        return np.concatenate(column_data_list, axis=1).astype(float)

    ###

    def inverse_transform_continuous(self, attr_transform_info, attr_data, sigmas, st):
        gm = attr_transform_info.transformer
        valid_component_indicator = attr_transform_info.transform_aux

        sel_normalized_value = attr_data[:, 0]
        sel_component_probs = attr_data[:, 1:]

        if sigmas is not None:
            sig = sigmas[st]
            sel_normalized_value = np.random.normal(sel_normalized_value, sig)

        sel_normalized_value = np.clip(sel_normalized_value, -1, 1)
        comp_probs = np.ones((len(attr_data), self._n_modes)) * -100
        comp_probs[:, valid_component_indicator] = sel_component_probs

        means = gm.means_.reshape([-1])
        stds = np.sqrt(gm.covariances_).reshape([-1])
        selected_component = np.argmax(comp_probs, axis=1)

        std_t = stds[selected_component]
        mean_t = means[selected_component]
        column = sel_normalized_value * 4 * std_t + mean_t

        return column

    def inverse_transform_discrete(self, attr_transform_info, attr_data):
        ohe = attr_transform_info.transformer
        return ohe.reverse_transform(attr_data)

    def inverse_transform(self, data, sigmas=None):
        """
        take matrix data and output raw data.
        is_dataframe: pd is_dataframe else numpy array
        """
        st = 0
        attr_data_list = []
        column_names = []
        for attr_transform_info in self._attr_transform_info:
            dim = attr_transform_info.output_dimensions
            column_data = data[:, st:st + dim]

            if attr_transform_info.column_type == 'continuous':
                attr_data = self.inverse_transform_continuous(
                    attr_transform_info, column_data, sigmas, st)
            else:
                assert attr_transform_info.column_type == 'discrete'
                attr_data = self.inverse_transform_discrete(
                    attr_transform_info, column_data)

            attr_data_list.append(attr_data)
            column_names.append(attr_transform_info.column_name)
            st += dim

        _data = np.column_stack(attr_data_list)
        _data = (pd.DataFrame(_data, columns=column_names).astype(self._attr_dtypes))
        if not self.is_dataframe:
            _data = _data.values

        return _data

    def convert_attr_name_val_to_id(self, attr_name, value):
        discrete_counter = 0
        column_id = 0
        for attr_transform_info in self._attr_transform_info:
            if attr_transform_info.column_name == attr_name:
                break
            if attr_transform_info.column_type == "discrete":
                discrete_counter += 1

            column_id += 1

        else:
            raise ValueError(f"The column_name `{attr_name}` does not exist in the data.")

        one_hot = attr_transform_info.transform.transform(np.array([value]))[0]
        if sum(one_hot) == 0:
            raise ValueError(f"The value `{value}` does not exist in the column `{attr_name}`.")

        return {
            "discrete_col_id": discrete_counter,
            "column_id": column_id,
            "value_id": np.argmax(one_hot)
        }
