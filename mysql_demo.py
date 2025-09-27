from pymysql import Connection

conn = Connection(
    host="localhost",
    port=3306,
    user="root",
    password="123456"
)

print(conn.get_server_info())

cursor = conn.cursor()
conn.select_db("mybatis-example")
cursor.execute("select * from schedule")

results: tuple = cursor.fetchall()
for item in results:
    print(item)

conn.close()
