import pandas as pd

data_paths = [
    "data/blind.test.txt",
    # "data/fake.test.txt",
    "data/fake.train.txt",
    "data/fake.valid.txt",
    "data/mix.test.txt",
    "data/mix.train.txt",
    "data/mix.valid.txt",
    "data/placeholder",
    "data/real.test.txt",
    "data/real.train.txt",
    "data/real.valid.txt"
]

'''
writes data into array
'''
def read_data(path, mixed=False, clean=False):
    labels = "See name of file"
    empties = []
    with open(path, "r") as f:
        data = f.read().split("<start_bio>")

    if clean:
        for d in range(len(data)):
            data[d] = data[d].replace("<end_bio>", "")
            data[d] = data[d].replace("\n", "")

            data[d] = data[d].strip()
            if data[d] == "":
                empties.append(d)

            #should we do something about the == xxxx === headers?
            #what about the labels, should they stay in or be a separate column

    if mixed:
        labels = ["" for d in data]
        for d in range(len(data)):
            if "[REAL]" in data[d]:
                data[d] = data[d].replace("[REAL]", "")
                labels[d] = "REAL"
            elif "[FAKE]" in data[d]:
                data[d] = data[d].replace("[FAKE]", "")
                labels[d] = "FAKE"
            else:
                labels[d] = "UNKNOWN"

    for e in empties:
        del data[e]
        del labels[e]
    
    assert len(data) == len(labels)

    return data, labels
    

read_data(data_paths[3], clean=True, mixed=True)