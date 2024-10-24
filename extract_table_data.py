import os
import re
import numpy as np
import pandas as pd
import sqlite3
import pickle


raw_data_file = "Data/unfun_2023-02-02.sql"
outputed_readable_file = "unfun_2023-02-02.sql"
#SQL_comand_PKL_file_name = "SQL_commands.pkl"
outputed_db_file_name = "Unfunned_data.db"
failed_comand_file_name = "failed_comands.pkl"
output_filtered_csv = True
filtered_csv_file_name = "unfun_2023.csv"

# Removing unreadable lines from the data
input_file = open(raw_data_file, 'r') 
output_file = open(outputed_readable_file, "w")
while True:
    try:
        line = input_file.readline()
        output_file.write(line)
        if line == '':
            break
    except:
        pass
input_file.close() 
output_file.close()

def convert_mysql_to_sqlite(mysql_sql_file, sqlite_sql_file):
    # Open the MySQL .sql file for reading
    with open(mysql_sql_file, 'r') as mysql_file:
        mysql_sql = mysql_file.read()

    # Step 1: Replace MySQL-specific data types with SQLite equivalents
    # Common conversions
    conversions = {
        r'\bTINYINT\b': 'INTEGER',
        r'\bSMALLINT\b': 'INTEGER',
        r'\bMEDIUMINT\b': 'INTEGER',
        r'\bINT\b': 'INTEGER',
        r'\bBIGINT\b': 'INTEGER',
        r'\bFLOAT\b': 'REAL',
        r'\bDOUBLE\b': 'REAL',
        r'\bDECIMAL\b': 'REAL',
        r'\bVARCHAR\(\d+\)\b': 'TEXT',
        r'\bCHAR\(\d+\)\b': 'TEXT',
        r'\bTEXT\b': 'TEXT',
        r'\bBLOB\b': 'BLOB',
        r'\bDATETIME\b': 'TEXT',  # SQLite stores dates as TEXT (ISO format recommended)
        r'\bAUTO_INCREMENT\b': '',  # Remove AUTO_INCREMENT, handled by INTEGER PRIMARY KEY in SQLite
        r'`': '"',  # Replace MySQL backticks with SQLite-compatible double quotes
    }

    # Apply the conversions
    for mysql_syntax, sqlite_syntax in conversions.items():
        mysql_sql = re.sub(mysql_syntax, sqlite_syntax, mysql_sql)

    # Step 2: Replace MySQL-specific SQL like `AUTO_INCREMENT`
    # Convert PRIMARY KEY with AUTO_INCREMENT to SQLite's `INTEGER PRIMARY KEY`
    mysql_sql = re.sub(r'INTEGER PRIMARY KEY', 'INTEGER PRIMARY KEY AUTOINCREMENT', mysql_sql, flags=re.IGNORECASE)

    # Step 3: Remove any engine or charset specifications in CREATE TABLE statements
    mysql_sql = re.sub(r'ENGINE=\w+', '', mysql_sql)
    mysql_sql = re.sub(r'DEFAULT CHARSET=\w+', '', mysql_sql)

    # Step 4: Remove comments or unsupported commands (optional)
    mysql_sql = re.sub(r'--.*?\n', '', mysql_sql)  # Remove single-line comments
    mysql_sql = re.sub(r'/\*.*?\*/', '', mysql_sql, flags=re.DOTALL)  # Remove block comments

    # Step 5: Write the transformed SQL to a new SQLite-compatible file
    with open(sqlite_sql_file, 'w') as sqlite_file:
        sqlite_file.write(mysql_sql)

    print(f"Conversion complete. SQLite-compatible .sql file saved as {sqlite_sql_file}")

convert_mysql_to_sqlite("unfun_2023-02-02.sql", "unfun_2023-02-02.sql")

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
with open(outputed_readable_file, 'r') as file:
    for _ in range(10):
        next(file)
        idx+=1
        print(idx,end="\r")
    for line in file:
        if line[:40].find("CREATE TABLE \"headlines_original\"") != -1:
            cursor = 0
        elif line[:40].find("INSERT INTO \"headlines_original\"") != -1:
            cursor = 1
        elif line[:40].find("CREATE TABLE \"headlines_unfunned\"") != -1:
            cursor = 2
        elif line[:40].find("INSERT INTO \"headlines_unfunned\"") != -1:
            cursor = 3
        elif line[:40].find("CREATE TABLE \"ratings\"") != -1:
            cursor = 4
        elif line[:40].find("INSERT INTO \"ratings\"") != -1:
            cursor = 5
        elif line == "\n":
            cursor = len(cursor_list)
        if cursor != len(cursor_list):
            commands[cursor_list[cursor]].append(line)
        idx+=1
        print(f"% {idx} | O: {len(commands["lines_original"])} | U: {len(commands["lines_unfunned"])} | R: {len(commands["lines_ratings"])}",end="\r")
print("\n")

# Saving outputed dictionary      
#with open(SQL_comand_PKL_file_name, "wb") as f:
#    pickle.dump(commands, f)

#with open('tables.pkl', 'rb') as f:
#    commands = pickle.load(f) # deserialize using load()

# Create an sql .db file and fill it with extracted commands
errors = {"lines_original": [],
          "lines_unfunned": [],
          "lines_ratings": []}
conn = sqlite3.connect(outputed_db_file_name)
cursor = conn.cursor()
nb_errors = 0
for table in np.array(list(commands.keys()))[["create" in string for string in list(commands.keys())]]:
    try:
        cursor.executescript(("".join(commands[table])).replace("ENGINE=InnoDB DEFAULT CHARSET=utf8", "").replace("\n",""))
        print("Table created successfully.")
    except sqlite3.Error as e:
        print(f"An error occurred when creating table: {e}")
for lines in np.array(list(commands.keys()))[["line" in string for string in list(commands.keys())]]:
        inserts_comands = ("".join(commands[lines])).split(");")
        for insert, idx in zip(inserts_comands, range(len(inserts_comands))):
            try:
                cursor.executescript((insert+");").replace("\\'","''").replace('\\"', '"'))
                #print("Script executed successfully.")
            except sqlite3.Error as e:
                nb_errors+=1
                errors[lines].append((insert+");").replace("\\'","''").replace('\\"', '"'))
                #errors[lines].append([idx, re.search('"(.+?)"', str(e)).group(1)])
                #print(f"An error occurred: during {lines} input, in INSERT: {idx}, {e}")
                #print((insert+";").replace("\\'","''").replace('\\"', '"'))
print("\n##############################################################\n")
print(f"Total number of unexecuted INSERT comands: {nb_errors}")
print(f"- original unexecuted INSERT: {len(errors["lines_original"])}")
print(f"- unfunned unexecuted INSERT: {len(errors["lines_unfunned"])}")
print(f"- ratings unexecuted INSERT: {len(errors["lines_ratings"])}")
#print(f"Saving dictionary of failed commands to: {failed_comand_file_name}\n\n")
## Saving failed commands   
#with open(failed_comand_file_name, "wb") as f:
#    pickle.dump(errors, f)

print("\n##############################################################\n")
print("Atempting to load errors chunks individualy")

insert_header: str = ""
individual_errors = {"lines_original": [],
                     "lines_unfunned": [],
                     "lines_ratings": []}
counter_successful = 0
counter_failed = 0
for tab_label in errors:
    for full_comand in errors[tab_label]:
        for line in full_comand.split("\n"):
            if line == "":
                continue
            elif line[:6] == "INSERT":
                insert_header = line
            else:
                try:   
                    cursor.execute(insert_header + " " + line[:-1] + ";")
                    counter_successful+=1
                except:
                    individual_errors[tab_label].append(insert_header + " " + line[:-1] + ";")
                    counter_failed+=1
    print(f"For table: {tab_label}, Successful: {counter_successful}, Failed: {counter_failed}")

#print(f"Saving dictionary of failed commands to: {failed_comand_file_name}\n\n")
# Saving failed commands   
#with open(failed_comand_file_name, "wb") as f:
#    pickle.dump(individual_errors, f)

if output_filtered_csv:
    request = """ SELECT DISTINCT O.id, O.title, U.title, R.rating, U.date
                  FROM headlines_original O, headlines_unfunned U, ratings R
                  WHERE O.id = U.original_headline_id 
                  and R.id = O.id
                  and U.title is not null;
                  """
    outputed_csv = pd.read_sql(request, con= conn).rename(columns={"title":"original_tile", "title.1":"unfuned_title"})
    #print(f"col names: {outputed_csv.columns}")
    replacement_dict = {"â€˜" : "'",
                        "â€™" : "'"}

    print("\nCleaning up strage characters")

    for i in range(len(outputed_csv["id"])):
        for key in replacement_dict:
            try:
                outputed_csv.iloc[i,1] = outputed_csv.iloc[i,1].replace(key,replacement_dict[key])
            except:
                pass
            try:
                outputed_csv.iloc[i,2] = outputed_csv.iloc[i,2].replace(key,replacement_dict[key])
            except:
                pass
        print(f"{round((i*100)/len(outputed_csv["id"]))}%",end="\r")

    outputed_csv.to_csv(filtered_csv_file_name, index=False)

    

conn.commit()
conn.close()
os.remove(outputed_readable_file)
os.remove(outputed_db_file_name)