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


class SeriesDataset(torch.utils.data.Dataset):
    def __init__(self, covariates, target, horizon=1, lagged_window=0, skip_interval=0):
        assert len(covariates) == len(target)
        self.covariates = torch.Tensor(covariates)
        self.target = torch.Tensor(target)
        self.lagg_window = lagged_window
        self.horizon = horizon
        self.skip_interval = skip_interval

    def __getitem__(self, index):
        """Get item at index
        Examples:
        >>> test = [ list(x) for x in zip(range(5), range(5))]
        >>> target = list(range(5, 10))
        >>> ds = SeriesDataset(test, target, horizon=2, lagged_window=1)
        >>> print(ds.__getitem__(0))
        {'lagged_numerical_covariates': tensor([[0., 0.]]), 'lagged_targets': tensor([5.]), 'covariates': tensor([[1., 1.],
        [2., 2.]]), 'target': tensor([6., 7.])}
        """
        # relative_idx = torch.arange(-self.lagg_window)
        lagged_covariates = self.covariates[index : index + self.lagg_window]
        lagged_targets = self.target[index : index + self.lagg_window]
        covariates = self.covariates[
            index + self.lagg_window : index + self.lagg_window + self.horizon
        ]
        target = self.target[
            index + self.lagg_window : index + self.lagg_window + self.horizon
        ]
        return {
            "lagged_numerical_covariates": lagged_covariates,
            "lagged_targets": lagged_targets,
            "covariates": covariates,
            "target": target,
        }

    def __len__(self):
        """
        Examples:
        >>> dataset = SeriesDataset(list(range(5)), list(range(5)))
        >>> len(dataset)
            5
        >>> dataset = SeriesDataset(list(range(5)), list(range(5)), horizon=2)
        >>> len(dataset)
            4
        >>> dataset = SeriesDataset(list(range(5)), list(range(5)), horizon=2, lagged_window=1)
        >>> len(dataset)
            3
        """
        return len(self.target) - self.lagg_window - self.horizon + 1

    @classmethod
    def from_dataframe(df, covariate_cols, target_cols):
        raise NotImplementedError()


if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
