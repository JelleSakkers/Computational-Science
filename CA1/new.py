def detect_cycle_and_length(values):
    seen = {}
    cycles = []

    for index, value in enumerate(values):
        if value in seen:
            # Cycle detected
            cycle_start = seen[value]
            cycle_length = index - cycle_start
            cycles.append((cycle_start, cycle_length))
        else:
            seen[value] = index

    if cycles:
        return True, cycles
    else:
        return False, []

# Example usage
my_list = ['011', '011', '110', '110', '011', '101']
has_cycle, cycles = detect_cycle_and_length(my_list)

if has_cycle:
    print("Cycles detected:")
    for cycle_start, cycle_length in cycles:
        print(f"Cycle starting at position {cycle_start}, with a length of {cycle_length}.")
else:
    print("No cycle detected.")

