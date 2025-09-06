import sqlite3
import json

def main():
    con = sqlite3.connect("nodes_all.db")
    cur = con.cursor()
    """
    cur.execute("CREATE TABLE node_all (id STRING PRIMARY KEY, data TEXT)")
    insertStmt = "INSERT INTO node_all (id, data) VALUES (?, ?)"

    # Create the table with all the nodes
    with open('../datasets/Twibot-20/node_lines.json', 'r') as file:
        for line in file:
            dataObj = json.loads(line.strip())
            cur.execute(insertStmt, (dataObj['id'], line))
    con.commit()

    query = "SELECT * FROM node_all LIMIT 3"
    for row in cur.execute(query):
        print(row)

    """
    # Create the table with the selected nodes
    cur.execute("CREATE TABLE node (id STRING PRIMARY KEY, IDX INTEGER, split STRING, label STRING, data TEXT)")
    insertStmt = "INSERT INTO node (id, idx, split, label, data) VALUES (?, ?, ?, ?, ?)"
    query = "SELECT DATA FROM node_all WHERE id = ?"
    labelsDict = readLabelsDict('../datasets/Twibot-20/label.csv')
    idx = 0
    firstLine = True
    with open('../datasets/Twibot-20/split.csv', 'r') as file:
        for line in file:
            if firstLine:
                firstLine = False
                continue
            tokens = line.strip().split(',')
            id = tokens[0]
            split = tokens[1]
            data = cur.execute(query, (id,)).fetchone()[0]
            label = labelsDict.get(id)
            cur.execute(insertStmt, (id, idx, split, label, data))
            idx += 1
    cur.execute("DROP TABLE node_all")
    con.commit()
    cur.execute("VACUUM")

    query = "SELECT * FROM node LIMIT 3"
    for row in cur.execute(query):
        print(row)

    query = "SELECT MIN(idx), MAX(idx) FROM node"
    for row in cur.execute(query):
        print(row)

    con.close()

def readLabelsDict(filename):
    labelsDict = {}
    firstLine = True
    with open(filename, 'r') as file:
        for line in file:
            if firstLine:
                firstLine = False
                continue
            tokens = line.strip().split(',')
            id = tokens[0]
            label = tokens[1]
            labelsDict[id] = label
        return labelsDict

if __name__ == "__main__":
    main()