with open("reqs.txt", "r") as f:
    f = f.read().split(", ")
    for i in f:
        print(i, end=" ")