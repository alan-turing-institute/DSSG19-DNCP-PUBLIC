/*
 -For the tender semantic table, these includes all the variables available to describe 
  the tender when it first enters the system for review.
 -DNCP provided a column indicating whether the variables are there at the beginning and whether 
  the variables are likely to change. 
 -The variables below are those that are at the beginning and does not change. 
 *Note: even if tipo_procedimiento is marked as 'able to change' the number of changes are so tiny that 
  these can be ignored. 
*/

drop table if exists semantic.tenders;
create table semantic.tenders as
select process.id_llamado,
planificacion_slug,
convocatoria_slug,
convocante,
unidad_contratacion,
tipo_entidad,
categoria,
"_objeto_licitacion",
fecha_publicacion_planificacion,
"_moneda",
planificacion_url,
tipo_procedimiento, 
tipo_procedimiento_codigo,
amt_planned_real_guarani,
audit.reception_date,
date_part('year', audit.reception_date) as reception_year,
date_part('month', audit.reception_date) as reception_month,
date_part('day', audit.reception_date) as reception_day,
case when amt_planned_real_guarani=0
then 0
else log(amt_planned_real_guarani) end as log_amt_planned_real_guarani
from cleaned.proceso_initial_cleaned as process
left join ( select id_llamado, 
			min(reception_date) as reception_date
			from cleaned.tender_records_cleaned
			group by id_llamado) audit
on process.id_llamado = audit.id_llamado;

delete from semantic.tenders
where reception_date is null;
--There are some unmatched first review 
--select count(distinct id_llamado) from raw.tender_record;
--select count(distinct id_llamado) from raw.proceso;
--select count(*)-count(reception_date), count(reception_date) from semantic.tenders;
--Tender records to be fixed by David's patch

--check if any of the missing values have an effective complaint
--with temp_table as (
--	select reception_date, 
--	labs.bool_of_effective_complaints as complaints,
--	reception_date is null as missing
--	from semantic.tenders as tenders
--	left join raw_labeled.tender_additional_info as labs
--	on tenders.id_llamado = labs.id_llamado
--)
--select missing, complaints, count(*)
--from temp_table
--group by missing, complaints;
--none of the missing reception values have an effective complaint









