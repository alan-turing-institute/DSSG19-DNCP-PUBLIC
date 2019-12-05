/*
 Creates production schema.
 For each object (agency, product type, etc.), create a table with the latest information available.
 Also creates support tables for tender value normalization.
*/

drop schema if exists production cascade;
create schema production;

drop table if exists production.agencies;
create table production.agencies as (
	select distinct on (convocante) *
	from semantic.tenders_agencies
	where date is not null 
	order by convocante, date desc
);

alter table production.agencies
drop column if exists id_llamado,
drop column if exists date;


drop table if exists production.agency_type;
create table production.agency_type as (
	select distinct on (tipo_entidad) *
	from semantic.tenders_agency_type
	where date is not null 
	order by tipo_entidad, date desc
);

alter table production.agency_type
drop column if exists id_llamado,
drop column if exists date;


drop table if exists production.procurement_type;
create table production.procurement_type as (
	select distinct on (tipo_procedimiento_codigo) *
	from semantic.tenders_procurement_type
	where date is not null 
	order by tipo_procedimiento_codigo, date desc
);

alter table production.procurement_type
drop column if exists id_llamado,
drop column if exists date;


drop table if exists production.product_type;
create table production.product_type as (
	select distinct on (categoria) *
	from semantic.tenders_product_type
	where date is not null 
	order by categoria, date desc
);

alter table production.product_type
drop column if exists id_llamado,
drop column if exists date;


drop table if exists production.service_type;
create table production.service_type as (
	select distinct on (_objeto_licitacion) *
	from semantic.tenders_service_type
	where date is not null 
	order by _objeto_licitacion, date desc
);

alter table production.service_type
drop column if exists id_llamado,
drop column if exists date;


drop table if exists production.inflation;
create table production.inflation as (
	select *
	from support_docs.inflation
);

drop table if exists production.exchange_rate;
create table production.exchange_rate as (
	select *
	from support_docs.exchange_rate
);
