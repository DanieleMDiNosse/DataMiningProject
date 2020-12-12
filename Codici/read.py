file = open("C:/Users/raffy/Desktop/temp/DataMiningProject/Codici/df.txt", "r")
term = "sil: 0.35"
clus = "clusters: 3"
for line in file:
    # line.strip().split("\n")
    if term in line and clus in line:
        print(line)
file.close()