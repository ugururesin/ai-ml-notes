import numpy as np
import pandas as pd
import mysql.connector
import urllib.request

def load_data_from_url(url):
    try:
        # load data from a URL
        data = urllib.request.urlopen(url)
        return data
    except Exception as e:
        print("Error while loading data from URL:", e)
        return None

def load_data_from_db(host, database, user, password, query):
    try:
        # load data from a database
        connection = mysql.connector.connect(
            host=host,
            database=database,
            user=user,
            password=password
        )
        cursor = connection.cursor()
        cursor.execute(query)
        result = cursor.fetchall()
        return result
    except Exception as e:
        print("Error while loading data from database:", e)
        return None

def load_data_from_local(filepath):
    try:
        # load data from a local directory
        data = np.loadtxt(filepath, delimiter=',')
        return data
    except Exception as e:
        print("Error while loading data from local directory:", e)
        return None

# Example usage
# data = load_data_from_url("https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv")
# data = load_data_from_db(host="hostname", database="dbname", user="user", password="password", query="SELECT * FROM tablename")
# data = load_data_from_local("data.csv")
