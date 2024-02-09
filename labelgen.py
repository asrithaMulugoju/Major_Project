
def LBGEN(n11,ntopics):
    map = []
    for i in range(1, ntopics+1):
        for j in range(n11*(i-1), n11*i):
            map.append(i)
    return map