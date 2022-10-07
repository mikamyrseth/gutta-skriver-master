rel = [23, 10, 33, 500, 70, 59, 82, 47, 72, 9]
ret = [55, 500, 2, 23, 72, 79, 82, 215]
ret_rel = set(rel).intersection(set(ret))

precission = len(ret_rel)/len(ret)
recall = len(ret_rel)/len(rel)

print("precission: ", precission)
print("recall: ", recall)


def get_best_precission(target_recall: float):
    best_precission = 0
    for i in range(len(ret)):
        ret2 = ret[0:i+1]
        # print(ret2)
        ret_rel = set(rel).intersection(set(ret2))
        # print(ret_rel)
        precission = len(ret_rel)/len(ret2)
        recall = len(ret_rel)/len(rel)
        if recall >= target_recall:
            best_precission = max(best_precission, precission)
    return best_precission


steps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for r in steps:
    print(r, get_best_precission(r))


print(range(0.0, 1.0, 0.1))
