--creates the exchange rate table yearly values from Paraguay central bank
--creates the inflation table yearly values from Paraguay central bank

create schema support_docs;

--create an inflation table

drop table if exists support_docs.inflation;
create table support_docs.inflation(
id_year int,
inflation_index real);

insert into support_docs.inflation(id_year, inflation_index)
values 
(2009,68.27242697532576),
(2010,71.44788869510836),
(2011,77.34517474613317),
(2012,80.18832070454319),
(2013,82.34046087343235),
(2014,86.48122075720205),
(2015,89.18722052837887),
(2016,92.8321441967672),
(2017,96.17640122391367),
(2018,100.0),
(2019,102.12222858545363);
--indices are based on 2018 values by setting 2018 values as 100 as basing other year values on that

--create exchange rate table
drop table if exists support_docs.exchange_rate;

create table support_docs.exchange_rate(
id_year int,
exchange_usd_guarani int);

insert into support_docs.exchange_rate(id_year, exchange_usd_guarani)
values 
 (2009, 4600),
 (2010, 4558),
 (2011, 4478),
 (2012, 4290),
 (2013, 4591),
 (2014, 4623),
 (2015, 5807),
 (2016, 5767),
 (2017, 5590),
 (2018, 5961),
 (2019, 6204);
