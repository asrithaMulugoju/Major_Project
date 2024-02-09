def vmap(n,RI,Tn):
    r = int(Tn/n)
    map = []
    for i in range(1,r+1):
        for j in range(0,Tn):
            if (RI[j]>=n*(i-1) and RI[j]<n*i):
                map.append(i)
    return map
