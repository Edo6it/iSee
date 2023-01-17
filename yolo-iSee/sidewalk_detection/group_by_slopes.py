

def get_groups(lines: list):
    #caso particolare, non ho linee, o ne ho solo una
    if len(lines) <= 1:
        return lines, []
    
    #caso generico, linee>=2
    groupA = []
    groupB = []

    #raggruppa le linee, sinistra e destra (non sai quale ma non importa)
    for line in lines:
        if (line.slope > 0):
            groupA.append(line)
        else:
            groupB.append(line)
        
    return groupA, groupB