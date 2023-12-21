import pandas as pd


def charge_data(dataset, optimizer):
    # Read and concatenate data using list comprehension
    print(f"Reading {dataset} {optimizer}...")
    files = [
        f"{dataset}/history_lin:3_conv:3_iter:{iter}_opt:{optimizer}_sch:False.csv"
        for iter in range(5)
    ]
    data = [
        pd.read_csv(file).assign(iter=iter, epoch=lambda df: df.index)
        for iter, file in enumerate(files)
    ]

    return pd.concat(data, ignore_index=True)


# create a new dataframe with the mean of the values as a row and mantain the columns
def get_mean_std(grouped, epoch):
    mean = grouped.drop(columns=["iter", "epoch"]).mean()
    mean = mean.to_frame().T

    std = grouped.drop(columns=["iter", "epoch"]).std()
    std = std.to_frame().T

    # Prefix std columns and concatenate mean and std dataframes
    std.columns = ["std_" + col for col in std.columns]
    result = pd.concat([mean, std], axis=1)
    result["epoch"] = epoch

    return result


def create_new_dataset(dataset, optimizer):
    data = charge_data(dataset, optimizer)

    data_grouped = data.groupby(["epoch"])
    data_grouped_0 = data_grouped.get_group((0))
    new_df = get_mean_std(data_grouped_0, 0)

    for i in range(1, 30):
        grouped = data_grouped.get_group((i))
        new_df = pd.concat([new_df, get_mean_std(grouped, i)])

    new_df["train_loss"] = new_df[" train_loss"]
    new_df["std_train_loss"] = new_df["std_ train_loss"]
    del new_df[" train_loss"]
    del new_df["std_ train_loss"]
    new_df.to_csv(f"Final/{dataset}_{optimizer}.csv", index=False)
    print(f"{dataset} {optimizer}.csv created\n")
