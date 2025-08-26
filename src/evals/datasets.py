from tqdm import tqdm
import openml
import pandas as pd
import os
from .. import BASE_DIR


# Find datasets with a certain number of features
def extend_datasets(datasets, filtering=False):
    extended_datasets = {}
    i = 0
    for d in tqdm(datasets):
        if (
                # Check if the dataset has the required information
            (not "NumberOfFeatures" in datasets[d])
                or (not "NumberOfClasses" in datasets[d])
                or (not "NumberOfInstances" in datasets[d])
                # or datasets[d]['NumberOfFeatures'] >= num_feats
                or datasets[d]["NumberOfClasses"] <= 0):
            # Skip datasets that don't meet the criteria
            print(datasets[d])
            continue
        ds = openml.datasets.get_dataset(d, download_data=False)
        if filtering and (
                # Apply additional filtering criteria
                datasets[d]["NumberOfInstances"] < 150
                or datasets[d]["NumberOfInstances"] > 2000
                or datasets[d]["NumberOfFeatures"] > 100
                or datasets[d]["NumberOfClasses"] > 10):
            # Skip datasets that don't meet the filtering criteria
            continue
        extended_datasets[d] = datasets[d]
        extended_datasets[d].update(ds.qualities)

    return extended_datasets


# Get a list of all datasets from OpenML
openml_list = openml.datasets.list_datasets()
openml_list = pd.DataFrame.from_dict(openml_list, orient="index")

# Select only classification datasets
openml_list = openml_list[~openml_list["MajorityClassSize"].isna()]

# Remove duplicated datasets based on specific columns
duplicated = openml_list.duplicated(
    subset=[
        "MajorityClassSize",
        "MaxNominalAttDistinctValues",
        "MinorityClassSize",
        "NumberOfClasses",
        "NumberOfFeatures",
        "NumberOfInstances",
        "NumberOfInstancesWithMissingValues",
        "NumberOfMissingValues",
        "NumberOfNumericFeatures",
        "NumberOfSymbolicFeatures",
    ],
    keep="first",
)
openml_list = openml_list[~duplicated]

duplicated = openml_list.duplicated(subset=["name"], keep="first")
openml_list = openml_list[~duplicated]

# # Filter out datasets that don't have meta information or don't fulfill other criteria
openml_list = openml_list.to_dict(orient="index")
openml_list = pd.DataFrame.from_dict(extend_datasets(openml_list,
                                                     filtering=False),
                                     orient="index")

# Remove specific datasets based on name or other criteria
openml_list = openml_list[~openml_list.name.apply(lambda x: "autoUniv" in x)]
openml_list = openml_list[~openml_list.name.apply(lambda x: "fri_" in x)]
openml_list = openml_list[~openml_list.name.apply(lambda x: "FOREX" in x)]
openml_list = openml_list[~openml_list.name.apply(lambda x: "ilpd" in x)]
openml_list = openml_list[~openml_list.name.apply(lambda x: "car" in x)]
openml_list = openml_list[~openml_list.name.apply(lambda x: "pc1" in x)]

# Remove datasets that failed to load
openml_list = openml_list[~openml_list.did.apply(
    lambda x: x in {
        1065, 40589, 41496, 770, 43097, 43148, 43255, 43595, 43786, 41701, 278,
        480, 462, 285
    })]

# Drop datasets with too many missing values
openml_list = openml_list[openml_list.NumberOfInstancesWithMissingValues /
                          openml_list.NumberOfInstances < 0.1]

# Remove datasets with class skew or perfect autocorrelation
openml_list = openml_list[(openml_list.MinorityClassSize /
                           openml_list.MajorityClassSize) > 0.05]
openml_list = openml_list[openml_list.AutoCorrelation != 1]

# Remove datasets that are too easy based on a specific evaluation metric
openml_list = openml_list[openml_list.CfsSubsetEval_DecisionStumpAUC != 1]

# filter with instances less than 4000 and features less than 100
openml_list = openml_list[openml_list["NumberOfInstances"] <= 4000]
openml_list = openml_list[openml_list["NumberOfFeatures"] <= 100]
openml_list = openml_list[openml_list["NumberOfClasses"] <= 10]
openml_list = openml_list[openml_list["NumberOfInstances"] >= 100]

# Print the final number of datasets
print("No. of datasets finalized:", len(openml_list))

props = [
    "name",
    "NumberOfFeatures",
    "NumberOfSymbolicFeatures",
    "NumberOfInstances",
    "NumberOfClasses",
    "NumberOfMissingValues",
    "MinorityClassSize",
]
# Prepare the table for printing and save it to a CSV file
renamer = {
    "name": "Name",
    "NumberOfFeatures": "# Features",
    "NumberOfSymbolicFeatures": "# Categorical Features",
    "NumberOfInstances": "# Instances",
    "NumberOfMissingValues": "# NaNs",
    "NumberOfClasses": "# Classes",
    "MinorityClassSize": "Minority Class Size",
}
print_table = openml_list
print_table = print_table[props].copy()
print_table["id"] = print_table.index
print_table[props[1:]] = print_table[props[1:]].astype(int)
print_table = print_table.rename(columns=renamer)
print_table = print_table.sort_values(by=["# Instances", "# Features"],
                                      ascending=False)
print_table = print_table.reset_index(drop=True)

print_table.to_csv(os.path.join(BASE_DIR, "data/openml_list.csv"), index=False)

# print to latex
print_table.columns = [
    'Name', 'Feat.', 'Cat.', 'Inst.', 'Class.', 'NaNs', 'Minor. Class Size',
    'OpenML ID'
]

# # Truncate long names to ...
# print_table = print_table.apply(lambda x: x.str[:30] + "..."
#                                 if x.dtype == "object" else x)
# print(
#     print_table.to_latex(index=False,
#                          escape=True,
#                          column_format="lrrrrrrr",
#                          longtable=True,
#                          label='datasets',
#                          caption='List of datasets used in the experiments.',
#                          position='H'))
