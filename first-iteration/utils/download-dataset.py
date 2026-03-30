import kagglehub

dataset1 = "rouseguy/bankbalanced"
dataset2 = "prasad22/retail-transactions-dataset"

path = kagglehub.dataset_download(dataset2)

print("Path to dataset files:", path)