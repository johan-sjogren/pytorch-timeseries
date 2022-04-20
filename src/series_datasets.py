import torch
import pandas as pd


# TODO: Include relative index for the items
# TODO: Class which samples the targets
# TODO: Include data only known prior to forecast


class DummyData:
    time_col = "order_date"
    prediction_groups = ["groupa", "groupb"]
    targets = ["pred1", "pred2"]

    def get_group_data(self, start_date=None, end_date=None):
        columns = [self.time_col] + self.prediction_groups + self.targets
        data = [
            ["2021-10-10", "A", "1", 0.1, 0.2],
            ["2021-10-11", "A", "1", 0.2, 0.3],
            ["2021-10-10", "A", "2", 0.2, 0.3],
            ["2021-10-11", "A", "2", 0.1, 0.2],
            ["2021-10-11", "B", "2", 0.2, 0.3],
            ["2021-10-10", "B", "2", 0.2, 0.3],
        ]
        fc = pd.DataFrame(columns=columns, data=data)
        return fc

    def get__data(self, start_date=None, end_date=None):
        columns = [self.time_col] + self.prediction_groups + self.targets
        data = [
            ["2021-10-10", "A", "1", 0.1, 0.2],
            ["2021-10-11", "A", "1", 0.2, 0.3],
            ["2021-10-13", "A", "1", 0.1, 0.2],
        ]
        fc = pd.DataFrame(columns=columns, data=data)
        return fc


class BaseSeriesDataset(torch.utils.data.Dataset):
    def __init__(
        self, target, horizon=1, lagged_window=0, skip_interval=0, dtype=torch.float64
    ):
        self.target = torch.Tensor(target).to(dtype)
        self.lagg_window = lagged_window
        self.horizon = horizon
        self.skip_interval = skip_interval

    def __len__(self):
        """
        Examples:
        >>> dataset = BaseSeriesDataset(list(range(5)))
        >>> len(dataset)
            5
        >>> dataset = BaseSeriesDataset(list(range(5)), horizon=2)
        >>> len(dataset)
            4
        >>> dataset = BaseSeriesDataset(list(range(5)), horizon=2, lagged_window=1)
        >>> len(dataset)
            3
        """
        return len(self.target) - self.lagg_window - self.horizon + 1

    def __getitem__(self, index):
        """Get item at index
        Examples:
        >>> target = list(range(5, 10))
        >>> ds = BaseSeriesDataset(target, horizon=2, lagged_window=1)
        >>> print(ds.__getitem__(0))
        (tensor([5.]), tensor([6., 7.]))
        """
        if index >= len(self):
            raise IndexError
        # relative_idx = torch.arange(-self.lagg_window)
        lagged_targets = self.target[index : index + self.lagg_window]
        target = self.target[
            index + self.lagg_window : index + self.lagg_window + self.horizon
        ]
        return target, lagged_targets


class SeriesWithCovariates(torch.utils.data.Dataset):
    def __init__(self, target, covariates, horizon=1, lagged_window=0, skip_interval=0):
        assert len(covariates) == len(target)
        self.covariates = BaseSeriesDataset(
            covariates, horizon, lagged_window, skip_interval
        )
        self.targets = BaseSeriesDataset(target, horizon, lagged_window, skip_interval)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        """Get item at index
        Examples:
        >>> test = [ list(x) for x in zip(range(5), range(5))]
        >>> target = list(range(5, 10))
        >>> ds = SeriesWithCovariates(target, test, horizon=2, lagged_window=1)
        >>> print(ds[0]['lagged_covariates'])
        tensor([[0., 0.]])
        >>> print(ds[0]['lagged_targets'])
        tensor([5.])
        >>> print(ds[0]['covariates'])
        tensor([[1., 1.],
                [2., 2.]])
        >>> print(ds[0]['targets'])
        tensor([6., 7.])
        """
        covariates, lagged_covariates = self.covariates[index]
        target, lagged_targets = self.targets[index]
        return {
            "lagged_covariates": lagged_covariates,
            "covariates": covariates,
            "lagged_targets": lagged_targets,
            "targets": target,
        }

    @classmethod
    def from_dataframe(cls, df, covariate_cols, target_cols):
        raise NotImplementedError()


class SeriesWithCategoricals(BaseSeriesDataset):
    def __init__(
        self,
        target,
        num_covariates,
        cat_covariates,
        horizon=1,
        lagged_window=0,
        skip_interval=0,
    ):
        assert len(num_covariates) == len(target)
        assert len(cat_covariates) == len(target)
        self.num_covariates = BaseSeriesDataset(
            num_covariates, horizon, lagged_window, skip_interval
        )
        self.cat_covariates = BaseSeriesDataset(
            cat_covariates, horizon, lagged_window, skip_interval, dtype=torch.int64
        )
        self.targets = BaseSeriesDataset(target, horizon, lagged_window, skip_interval)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        """Get item at index
        Examples:
        >>> test = [ list(x) for x in zip(range(5), range(5))]
        >>> cats = [ list(x) for x in zip(range(5,10), range(6, 11))]
        >>> target = list(range(5, 10))
        >>> ds = SeriesWithCategoricals(target, test, cats, horizon=2, lagged_window=1)
        >>> print(ds.__getitem__(0)['lagged_categorical_covariates'])
        tensor([[5., 6.]])
        >>> print(ds[0]['lagged_numerical_covariates'])
        tensor([[0., 0.]])
        >>> print(ds[0]['lagged_targets'])
        tensor([5.])
        >>> print(ds[0]['categorical_covariates'])
        tensor([[6., 7.],
                [7., 8.]])
        >>> print(ds[0]['numerical_covariates'])
        tensor([[1., 1.],
                [2., 2.]])
        >>> print(ds[0]['targets'])
        tensor([6., 7.])
        """

        cat_covariates, lagged_cat_covariates = self.cat_covariates[index]
        num_covariates, lagged_num_covariates = self.num_covariates[index]
        target, lagged_targets = self.targets[index]
        return {
            "lagged_categorical_covariates": lagged_cat_covariates,
            "categorical_covariates": cat_covariates,
            "lagged_numerical_covariates": lagged_num_covariates,
            "numerical_covariates": num_covariates,
            "lagged_targets": lagged_targets,
            "targets": target,
        }

    @classmethod
    def from_dataframe(cls, df, covariate_cols, target_cols):
        raise NotImplementedError()


class GroupedSeriesDS(torch.utils.data.Dataset):
    def __init__(
        self, group_dict, dataset_type, horizon=1, lagged_window=0, skip_interval=0
    ):
        """
        >>> group_dict = {'a':[list(range(5))], 'b': [list(range(5, 10))]}
        >>> ds = GroupedSeriesDS(group_dict, BaseSeriesDataset, horizon=2, lagged_window=1)
        >>> len(ds)
        6
        >>> ds.group_dict['b'][0]
        (tensor([5.]), tensor([6., 7.]))
        >>> ds[3]
        (tensor([5.]), tensor([6., 7.]))
        >>> ds[0]
        (tensor([0.]), tensor([1., 2.]))
        """
        self.group_dict = {}
        self.idx_dict = {}
        _idx = 0
        for group, data in group_dict.items():
            self.group_dict[group] = dataset_type(
                *data, horizon, lagged_window, skip_interval
            )
            temp_dict = {
                idx: (_idx, group)
                for idx in range(_idx, _idx + len(self.group_dict[group]))
            }
            _idx += len(self.group_dict[group])
            self.idx_dict.update(temp_dict)
        self.idx_max = _idx

    def __len__(self):
        return self.idx_max

    def __getitem__(self, index):
        idx, group = self.idx_dict[index]
        return self.group_dict[group][index - idx]

    @classmethod
    def from_dataframe(
        cls,
        df,
        group_cols,
        target_cols,
        num_covariate_cols=None,
        cat_covariate_cols=None,
        keep_groups=False,
        **kwargs
    ):
        grouped = df.groupby(group_cols, as_index=not keep_groups)

        if keep_groups:
            if cat_covariate_cols:
                cat_covariate_cols += group_cols
            else:
                cat_covariate_cols = group_cols

        group_dict = {}
        if (num_covariate_cols is None) and (cat_covariate_cols is None):
            for group, data in grouped:
                group_dict[group] = [data[target_cols].values]
            return cls(group_dict, BaseSeriesDataset, **kwargs)
        elif num_covariate_cols is None:
            for group, data in grouped:
                group_dict[group] = [
                    data[target_cols].values,
                    data[cat_covariate_cols].values,
                ]
            return cls(group_dict, SeriesWithCovariates, **kwargs)
        elif cat_covariate_cols is None:
            for group, data in grouped:
                group_dict[group] = [
                    data[target_cols].values,
                    data[num_covariate_cols].values,
                ]
            return cls(group_dict, SeriesWithCovariates, **kwargs)
        else:
            for group, data in grouped:
                group_dict[group] = [
                    data[target_cols].values,
                    data[num_covariate_cols].values,
                    data[cat_covariate_cols].values,
                ]
            return cls(group_dict, SeriesWithCategoricals, **kwargs)


class TimeSeriesDataset(BaseSeriesDataset):
    def __init__(
        self,
        target,
        num_covariates=None,
        cat_covariates=None,
        horizon=1,
        lagged_window=0,
        skip_interval=0,
    ):
        assert len(cat_covariates) == len(target)
        super().__init__(num_covariates, target, horizon, lagged_window, skip_interval)


class HierarchialSeriesDS(torch.utils.data.Dataset):
    pass
