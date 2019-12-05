**1. Creating database**

We create a new database called **_dncp_** and run the _20190703 restore.sql_ given to us by DNCP. This creates a new schema called _dsfsg_. We later rename this to _raw_.

**2. Create a copy of dump file, just in case** (This is named _restore2.sql_)

**3. Change owner name to _dbadmin_ to meet our requirements.**

Since postgres user does not exist, we change the owner of the tables to _dbadmin_ for whom we have the credentials.

```
cd ./data
sed 's/OWNER TO postgres;/OWNER TO dbadmin;/g' restore2.sql > restore2_mod.sql
```

**4. Load SQL dumps**
To load the data, we need to connect to a DB (in this case, _dncp_) and from there launch the corresponding SQL dumps in order.

```
psql service=postgres
create database dncp
\c dncp
>> \i restore2_mod.sql
>> alter schema dsfsg rename to raw
```

All good!! Check the database for data! :smile:
