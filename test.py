import pickle


# Extracting all sql commands for relevent tables
commands = {
"create_original" : [],
"lines_original" : [],
"create_unfunned" : [],
"lines_unfunned" : [],
"create_ratings" : [],
"lines_ratings" : []
}
cursor_list = list(commands.keys())
cursor = len(cursor_list)
idx = 0
with open("unfun_2023-02-02_test.sql", 'r') as file:
    for _ in range(10):
        next(file)
        idx+=1
        print(idx,end="\r")
    for line in file:
        if line.find("CREATE TABLE \"headlines_original\"") != -1:
            cursor = 0
        elif line.find("INSERT INTO \"headlines_original\"") != -1:
            cursor = 1
        elif line.find("CREATE TABLE \"headlines_unfunned\"") != -1:
            cursor = 2
        elif line.find("INSERT INTO \"headlines_unfunned\"") != -1:
            cursor = 3
        elif line.find("CREATE TABLE \"ratings\"") != -1:
            cursor = 4
        elif line.find("INSERT INTO \"ratings\"") != -1:
            cursor = 5
        elif line == "\n":
            cursor = len(cursor_list)
        if cursor != len(cursor_list):
            commands[cursor_list[cursor]].append(line)
        idx+=1
        print(f"% {idx} | O: {len(commands["lines_original"])} | U: {len(commands["lines_unfunned"])} | R: {len(commands["lines_ratings"])}",end="\r")
print("\n")

# Saving outputed dictionary      
with open("new_tab_out.pkl", "wb") as f:
    pickle.dump(commands, f)