drop schema if exists raw_labeled cascade;
create schema raw_labeled;


/* Create new information
 *
 * UPDATE JUST THIS TABLE!!
 * */

drop table if exists raw_labeled.tender_additional_info;

create table raw_labeled.tender_additional_info as
with complaint_fecha as (
	select id_llamado
			, min(fecha_resolucion_proceso) min_fecha
	from raw.proceso_juridico
	group by id_llamado
)
select
	tender.id_llamado
	, count(distinct complaint.denunciante) number_of_total_complainers -- number of total complainers
	, count(complaint.*) number_of_total_complaints -- number of total complaints
	, sum(case when complaint."_tipo_resultado" = 'a_favor' then 1 else 0 end) number_of_effective_complaints -- number of effective complaints
	, max(case when complaint."_tipo_resultado" = 'a_favor' then 1 else 0 end) bool_of_effective_complaints -- bool of effective complaints
	, sum(case when documents.tipo_documento = 'Adenda' then 1 else 0 end) number_of_amendments-- number of amendments
 	, max(case when documents.tipo_documento = 'Adenda' then 1 else 0 end) bool_of_amendments-- bool of amendments
	, sum(case when documents.tipo_documento = 'Adenda'
		and tender.fecha_publicacion_convocatoria > documents.fecha_archivo
		then 1 else 0 end) number_of_amendments_0 -- number of amndments 0 (before publication)
	, max(case when documents.tipo_documento = 'Adenda'
		and tender.fecha_publicacion_convocatoria > documents.fecha_archivo
		then 1 else 0 end) bool_of_amendments_0 -- bool of amndments 0 (before publication)
	, sum(case when documents.tipo_documento = 'Adenda'
		and tender.fecha_publicacion_convocatoria < documents.fecha_archivo
		and documents.fecha_archivo < complaint_fecha.min_fecha
		then 1 else 0 end) number_of_amendments_1 -- number of amndments 1 (between publication and 1 complaint)
	, max(case when documents.tipo_documento = 'Adenda'
		and tender.fecha_publicacion_convocatoria < documents.fecha_archivo
		and documents.fecha_archivo < complaint_fecha.min_fecha
		then 1 else 0 end) bool_of_amendments_1 -- number of amndments 1 (between publication and 1 complaint)
	, sum(case when documents.tipo_documento = 'Adenda'
		and documents.fecha_archivo > complaint_fecha.min_fecha
		then 1 else 0 end) number_of_amendments_2 -- number of amndments 2 (after complaint)
	, max(case when documents.tipo_documento = 'Adenda'
		and documents.fecha_archivo > complaint_fecha.min_fecha
		then 1 else 0 end) bool_of_amendments_2 -- number of amndments 2 (after complaint)
	-- TODO: corrected values (@WEN) [correct all the columns please]
	, max(monto_global_estimado_planificacion) amt_planned_ori
	, max(monto_estimado_convocatoria) amt_tender_ori
	, max(monto_total_adjudicado) amt_award_ori
	, round(cast(max(monto_global_estimado_planificacion/(
			select inflation_index/100 from support_docs.inflation
			where id_year = extract(year from tender.fecha_publicacion_planificacion))*(
				select case when tender."_moneda" = 'USD'
				then exchange_usd_guarani
				else 1
				end
				from support_docs.exchange_rate
				where id_year = extract(year from tender.fecha_publicacion_planificacion))
				) as numeric),2) amt_planned_real_guarani
	, round(cast(max(monto_estimado_convocatoria/(
		select inflation_index/100 from support_docs.inflation
		where id_year = extract(year from tender.fecha_publicacion_convocatoria))*(
			select case when tender."_moneda" = 'USD'
			then exchange_usd_guarani
			else 1
			end
			from support_docs.exchange_rate
			where id_year = extract(year from tender.fecha_publicacion_convocatoria))
			) as numeric),2) amt_tender_real_guarani
	, round(cast(max(monto_total_adjudicado/(
	select inflation_index/100 from support_docs.inflation
	where id_year = extract(year from tender.fecha_publicacion_adjudicacion))*(
		select case when tender."_moneda" = 'USD'
		then exchange_usd_guarani
		else 1
		end
		from support_docs.exchange_rate
		where id_year = extract(year from tender.fecha_publicacion_adjudicacion))
		) as numeric),2) amt_awarded_real_guarani
from
	raw.proceso as tender
left join
	complaint_fecha
on tender.id_llamado = complaint_fecha.id_llamado
left join
	raw.proceso_juridico as complaint
on tender.id_llamado = complaint.id_llamado
left join
	raw.pbc_adenda as documents
on tender.id_llamado = documents.id_llamado
group by
	tender.id_llamado;

--Add a column to indicate price discrepancies between tender and planning.
drop table if exists raw_labeled.tempo;
create table raw_labeled.tempo as(
select
	*,(abs(amt_tender_real_guarani-amt_planned_real_guarani)/case
			when amt_planned_real_guarani>=amt_tender_real_guarani then amt_tender_real_guarani+1
			else amt_planned_real_guarani+1 --+1 is added to avoid division by zero error
		end)>6000 as high_difference
from raw_labeled.tender_additional_info);
drop table raw_labeled.tender_additional_info;
alter table raw_labeled.tempo rename to tender_additional_info;


--Check the impact of discrepancy where the difference between planned and tender are more than exchange rate
--with diff as (
--	select amt_tender_real_guarani, amt_planned_real_guarani, bool_of_effective_complaints,
--		(abs(amt_tender_real_guarani-amt_planned_real_guarani)/case
--			when amt_planned_real_guarani>=amt_tender_real_guarani then amt_tender_real_guarani+1
--			else amt_planned_real_guarani+1 --+1 is added to avoid division by zero error
--		end)>6000 as difference
--	from raw_labeled.tender_additional_info)
--select difference, bool_of_effective_complaints,count(*)
--from diff
--group by difference, bool_of_effective_complaints;

/*Updates the rest of the tables*/

drop table if exists raw_labeled.proceso;
create table raw_labeled.proceso as
select
*
from raw.proceso
natural left join raw_labeled.tender_additional_info ;

drop table if exists raw_labeled.pbc_adenda;
create table raw_labeled.pbc_adenda as
select
*
from raw.pbc_adenda
natural left join raw_labeled.tender_additional_info ;

--create new item table with cleaned prices to reflect exchange rate and inflation & additional info
/*
drop table if exists raw_labeled.item_solicitado;
create table raw_labeled.item_solicitado as
select
*, round(cast(monto/(
	select inflation_index/100 from support_docs.inflation
	where id_year = extract(year from item.fecha_publicacion))*(
		select case when item."_moneda" = 'USD'
		then exchange_usd_guarani
		else 1
		end
		from support_docs.exchange_rate
		where id_year = extract(year from item.fecha_publicacion))
		as numeric),2) amt_item_real_guaranis
from raw.item_solicitado as item
natural left join raw_labeled.tender_additional_info;
*/


drop table if exists raw_labeled.contrato;
create table raw_labeled.contrato as
select
*
from raw.contrato
natural left join raw_labeled.tender_additional_info ;
