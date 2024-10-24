from typing import List,Dict, Callable, Any
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import re


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

def clean_sql_file(input_file_path: str, output_file_path: str):
    input_file = open(input_file_path, 'r') 
    output_file = open(output_file_path, "w")
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

# Define a function to apply a function concurrently
def lapply(fun: Callable, array: list, *args) -> List[Any]:
    out = []
    with ThreadPoolExecutor(max_workers=len(array)) as executor:
        for i in range(len(array)):
            out.append(executor.submit(fun, array[i], *args))
        for i in range(len(out)):
            out[i] = out[i].result()
        return out
    
def filter_line(line: str):
    try:
        if line[-2] != ';' and line[-2] != ',':
            return ""
    except:
        return line
    finally:
        return line
    
ref_dict = {"Start_original" : 2,
            "Start_unfunned" : 3,
            "Start_ratings" : 4,
            "Entry" : 1,
            "Comand_end" : 5,
            "None" : 0}

def id_lines(line: str, ref_dict: Dict[str, int]) -> int:
    try:
        if line[0] == '(' and line[-2] == "," and line[-3] == ")":
            return ref_dict["Entry"]
        if line[-2] == ";":
            return ref_dict["Comand_end"]
        if line[:32] == "INSERT INTO `headlines_original`":
            return ref_dict["Start_original"]
        if line[:32] == "INSERT INTO `headlines_unfunned`":
            return ref_dict["Start_unfunned"]
        if line[:21] == "INSERT INTO `ratings`":
            return ref_dict["Start_ratings"]
        return ref_dict["None"]
    except:
        return ref_dict["None"]

def extract_lines(file_path: str) -> List[str]:
    lines = []
    with open(file_path, 'r') as file:
        while True:
            try:
                line = file.readline()
                lines.append(line)
                if line == '':  # Empty string means end of the file
                    break
            except:
                pass
    return lines

