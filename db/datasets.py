# import sys
# # Add the path to your config.py to sys.path
# sys.path.append(r'C:\Users\saksh\OneDrive\Desktop\stuffs\Chartreader-with-gpu\db')
# # Now you can import system_configs from config.py
# from coco import Chart

# dataset= {
#     "Chart": Chart
# }

def load_datasets():
    from coco import Chart
    return {
        "Chart": Chart
    }

# This function will be called to get the dataset
def get_dataset(dataset_name):
    datasets = load_datasets()
    if dataset_name in datasets:
        return datasets[dataset_name]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")