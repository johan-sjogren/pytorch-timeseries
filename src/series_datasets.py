import torch
import pandas as pd


# TODO: Include relative index for the items
# TODO: split covariates into numerical and categorical
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
    def __init__(self, target, horizon=1, lagged_window=0, skip_interval=0):
        self.target = torch.Tensor(target)
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
        >>> print(ds.__getitem__(0)['lagged_targets'])
        tensor([5.])
        >>> print(ds.__getitem__(0)['targets'])
        tensor([6., 7.])
        """
        # relative_idx = torch.arange(-self.lagg_window)
        lagged_targets = self.target[index : index + self.lagg_window]
        target = self.target[
            index + self.lagg_window : index + self.lagg_window + self.horizon
        ]
        return {
            "lagged_targets": lagged_targets,
            "targets": target,
        }


class SeriesDatasetWithCovariates(BaseSeriesDataset):
    def __init__(self, covariates, target, horizon=1, lagged_window=0, skip_interval=0):
        assert len(covariates) == len(target)
        self.covariates = torch.Tensor(covariates)
        super().__init__(target, horizon, lagged_window, skip_interval)

    def __getitem__(self, index):
        """Get item at index
        Examples:
        >>> test = [ list(x) for x in zip(range(5), range(5))]
        >>> target = list(range(5, 10))
        >>> ds = SeriesDatasetWithCovariates(test, target, horizon=2, lagged_window=1)
        >>> print(ds.__getitem__(0)['lagged_covariates'])
        tensor([[0., 0.]])
        >>> print(ds.__getitem__(0)['lagged_targets'])
        tensor([5.])
        >>> print(ds.__getitem__(0)['covariates'])
        tensor([[1., 1.],
                [2., 2.]])
        >>> print(ds.__getitem__(0)['targets'])
        tensor([6., 7.])
        """

        # relative_idx = torch.arange(-self.lagg_window)
        lagged_covariates = self.covariates[index : index + self.lagg_window]
        covariates = self.covariates[
            index + self.lagg_window : index + self.lagg_window + self.horizon
        ]
        return_dict = super().__getitem__(index)
        return_dict.update(
            {
                "lagged_covariates": lagged_covariates,
                "covariates": covariates,
            }
        )
        return return_dict

    @classmethod
    def from_dataframe(df, covariate_cols, target_cols):
        raise NotImplementedError()


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


class GroupedSeriesDS(torch.utils.data.Dataset):
    pass


class HierarchialSeriesDS(torch.utils.data.Dataset):
    pass
