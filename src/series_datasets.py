import torch
import pandas as pd

class DummyData:
    time_col = "order_date"
    prediction_groups = ["groupa", "groupb"]
    targets = ["pred1", "pred2"]

    def get_data(self, start_date=None, end_date=None):
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


class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, covariates, target, horizon=1, lagged_window=0):
        # TODO: Verify that the lengths match
        self.covariates = torch.Tensor(covariates)
        self.target = torch.Tensor(target)
        self.lagg_window = lagged_window
        self.horizon = horizon
        self.shape = self.__getshape__()
        self.size = self.__getsize__()

    def __getitem__(self, index):
        """ Get item at index
        """
        lagged_covariates = self.covariates[index:index+self.lagg_window]
        lagged_targets = self.target[index:index+self.lagg_window]
        covariates = self.covariates[index+self.lagg_window:index+self.lagg_window+self.horizon]
        target = self.target[index+self.lagg_window:index+self.lagg_window+self.horizon]
        return lagged_covariates, lagged_covariates, covariates, target

    def __len__(self):
        """
        Examples:
        >>> dataset = TimeSeriesDataset(list(range(5)), list(range(5)))
        >>> len(dataset)
            5
        >>> dataset = TimeSeriesDataset(list(range(5)), list(range(5)), horizon=2)
        >>> len(dataset)
            4
        >>> dataset = TimeSeriesDataset(list(range(5)), list(range(5)), horizon=2, lagged_window=1)
        >>> len(dataset)
            3
        """
        return len(self.target) - self.lagg_window - self.horizon + 1
    
    def __getshape__(self):
        return (self.__len__(), *self.__getitem__(0)[0].shape)
    
    def __getsize__(self):
        return (self.__len__())

    @classmethod
    def from_dataframe(df, covariate_cols, target_cols):
        raise NotImplementedError()

if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)