from models import ImplyingNodes

def conflict_jaccard(prev: ImplyingNodes, curr: ImplyingNodes) -> float:
    #Jaccard overlap of the (facet, dimension) conflist sets of two cycles
    #first cycle: identify conflict cells (fully stable)
    #disjoints ( if every conflict is new) 
    

    prev_set = {(c.facet, c.dimension) for c in prev.conflicts}
    curr_set = {(c.facet, c.dimension) for c in curr.conflicts}

    if not prev_set and not curr_set:
        return 1.0
    return len(prev_set & curr_set) / len(prev_set | curr_set)
