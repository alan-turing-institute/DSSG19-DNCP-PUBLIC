-- Create empty table to upload data from src/etl/get_regions.py script
create table semantic.demographics (
    departament varchar,
    municipality varchar,
    idh varchar
);

-- Merge convocante_slug(municipalities) with departments
create table semantic.tenders_demographics as
select 
tenders.convocante_slug,
replace(tenders.convocante_slug,'-',' '),
demographics.municipality,
demographics.departament,
demographics.idh
from cleaned.proceso_initial tenders
left join semantic.demographics demographics
on replace(tenders.convocante_slug,'-',' ') like concat('%',demographics.municipality,'%')
where tenders.tipo_entidad='Municipalidades';

-- Manually update unmatched
update semantic.tenders_demographics
set 
departament = 'san pedro',
municipality = 'san vicente pancholo'
where
convocante_slug = 'municipalidad-san-vicente-pancholo';

-- Add idh and id_llamado
-- drop table semantic.tenders_demographics_2;
create table semantic.tenders_demographics_2 as
select 
a.id_llamado,
a.convocante_slug,
b.municipality,
b.departament,
c.idh
from (select * from cleaned.proceso_initial where tipo_entidad='Municipalidades') a
inner join (select distinct (convocante_slug),municipality,departament from semantic.tenders_demographics) b
on a.convocante_slug = replace(b.convocante_slug,' ','-')
left join (select distinct (departament),idh from semantic.demographics)c
on b.departament = c.departament;

------------ sanity_check -----------

-- some id_llamado are duplicated as there are similar names for same municipalities, or some
-- have the same name but belong to different region. 

-- table has 40K tenders
select count(*) from semantic.tenders_demographics_2;
-- original table has 36K tenders
select count(*) from (select * from cleaned.proceso_initial where tipo_entidad='Municipalidades') a;
-- find out which are duplicated to understand the problem
select * from semantic.tenders_demographics_2 where id_llamado in
(select id_llamado from semantic.tenders_demographics_2 group by id_llamado having count(* ) > 1  )
order by id_llamado;

--- solve the problem

-- 1. find out convocante_slug that are duplicated
select distinct convocante_slug from semantic.tenders_demographics_2 where id_llamado in
(select id_llamado from semantic.tenders_demographics_2 group by id_llamado having count(* ) > 1  );

-- 2. see in details the convocante_slug that are duplicated
select * from semantic.tenders_demographics_2 where id_llamado in
(select id_llamado from semantic.tenders_demographics_2 group by id_llamado having count(* ) > 1  )
 and convocante_slug = 'municipalidad-mbocayaty-yhaguy'
order by id_llamado;

-- 3. change convocante_slug manually
update semantic.tenders_demographics_2
set 
departament = 'san pedro',
municipality = 'yataity norte',
idh = '0.695'
where
convocante_slug = 'municipalidad-yataity-norte';

-- 4. remove duplicates
drop table if exists semantic.tenders_demographics_3;
create table semantic.tenders_demographics_3 as
select distinct id_llamado,convocante_slug,municipality,departament,idh from semantic.tenders_demographics_2 ;

------------ sanity_check -----------
-- new table has 40K tenders
select count(*) from semantic.tenders_demographics_2;
select count(*) from semantic.tenders_demographics_3;
-- original table has 36K tenders
select count(*) from (select * from cleaned.proceso_initial where tipo_entidad='Municipalidades') a;
-- check all departaments have the same idh
select distinct departament,idh from semantic.tenders_demographics_3;


-- drop intermediate tables and rename final table
drop table semantic.tenders_demographics;
drop table semantic.tenders_demographics_2;
alter table  semantic.tenders_demographics_3 rename to tenders_demographics;


