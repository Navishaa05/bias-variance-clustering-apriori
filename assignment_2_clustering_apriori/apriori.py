from itertools import combinations

def Apriori(input_list, min_support, min_confidence):
    # Clean input: remove extra spaces but preserve duplicates
    transactions = []
    all_items = []

    for row in input_list:
        cleaned = [item.strip() for item in row]
        transactions.append(cleaned)
        all_items.extend(cleaned)

    total_records = len(transactions)

    # Count support for an itemset (as a list)
    def count_support(itemset):
        count = 0
        for t in transactions:
            # Check if all items in itemset are in the transaction (allow duplicates)
            t_copy = t[:]
            found = True
            for item in itemset:
                if item in t_copy:
                    t_copy.remove(item)
                else:
                    found = False
                    break
            if found:
                count += 1
        return count / total_records

    # First level: single items (including repeated ones)
    unique_items = sorted(set(all_items))
    level = []
    for item in unique_items:
        itemset = [item]
        if count_support(itemset) >= min_support:
            level.append(itemset)

    all_frequent = {1: level}
    all_supports = {tuple(fs): count_support(fs) for fs in level}

    k = 2
    while True:
        prev = all_frequent[k - 1]
        candidates = []

        size = len(prev)
        for i in range(size):
            for j in range(i + 1, size):
                l1 = prev[i]
                l2 = prev[j]
                if l1[:k - 2] == l2[:k - 2]:
                    union = l1 + [l2[-1]]
                    if count_support(union) >= min_support:
                        # Check all k-1 subsets of union are in prev
                        valid = True
                        for subset in combinations(union, k - 1):
                            if list(subset) not in prev:
                                valid = False
                                break
                        if valid:
                            candidates.append(union)

        if not candidates:
            break

        all_frequent[k] = candidates
        for fs in candidates:
            all_supports[tuple(fs)] = count_support(fs)
        k += 1

    # Association Rule Generation
    printed_rules = set()
    for itemset_tuple in all_supports:
        itemset = list(itemset_tuple)
        if len(itemset) < 2:
            continue

        n = len(itemset)
        for bit in range(1, 2 ** n - 1):
            left = []
            right = []
            for idx in range(n):
                if (bit >> idx) & 1:
                    left.append(itemset[idx])
                else:
                    right.append(itemset[idx])

            if left and right:
                lhs = tuple(left)
                rhs = tuple(right)
                full = tuple(itemset)
                if lhs in all_supports:
                    conf = all_supports[full] / all_supports[lhs]
                    if conf >= min_confidence:
                        lhs_str = ", ".join(lhs)
                        rhs_str = ", ".join(rhs)
                        rule = f"{lhs_str} −> {rhs_str}"
                        if rule not in printed_rules:
                            print(rule)
                            printed_rules.add(rule)


# # Sample usage
# if __name__ == "__main__":
#     input_list = [
#         [' A', 'B', 'C '],
#         [' D', 'E '],
#         [' A', 'D', 'F '],
#         [' E ', 'F '],
#         [' B', 'D', 'E '],
#         [' A', 'D'],
#         [' A', 'B', 'E ', 'D'],
#         [' C', 'F '],
#         [' A', 'B', 'D'],
#         [' D', 'E ']
#     ]
#     min_support = 0.2
#     min_confidence = 0.6

#     Apriori(input_list, min_support, min_confidence)
