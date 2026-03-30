import kagglehub

dataset1 = "rouseguy/bankbalanced"
dataset2 = "prasad22/retail-transactions-dataset"
dataset3 = "crawlfeeds/tesco-uk-groceries-dataset"
dataset4 = "vjchoudhary7/customer-segmentation-tutorial-in-python"

path = kagglehub.dataset_download(dataset4)

print("Path to dataset files:", path)