--------------------- Descriptive statistics on Complaints and Amendments -----------------------
-- In this query you can find tables that summarise data about:

-- 1. Amendments 1: Adendas related to tender process
-- 2. Complaints
-- 3. Amendments 2: Adendas related to complaints
-- 4. Sumarizing reviewing process, PBC (tender documents), Amendments 1 and amendments 2 at tender level

------------------1.  Amendments 1: Adendas related to tender process
------1.1. Reviewing
-- How many reviewing processes stages do they have?
select
	final_state,
	count(*)
from 
	raw.tender_record 
group by 1;

-- Quanty of tenders vs Quantity of tenders that were reviewed
select
	count(distinct tenders.id_llamado) quantity_tenders,
	count(distinct review.llamado_id) quantity_tenders_reviewed
from
	raw.proceso tenders
left join
	raw.tender_record review
on tenders.id_llamado = review.llamado_id;

-- Are the tenders that were not reviewed still open? Why aren't they reviewed?
-- No! There are 25K tenders that were not reviewed but they are awarded.
select
	tenders.etapa_licitacion,
	count(*)
from
	raw.proceso tenders
left join
	raw.tender_record review
on tenders.id_llamado = review.llamado_id
where review.llamado_id is null
group by 1
order by 2 desc;

------- 1.2 About documents table (pbc_adenda)

-- How many tipo documento we have?
select 
	amendments.tipo_documento,
	count(distinct  tenders.id_llamado)
from raw.proceso tenders
inner join raw.pbc_adenda amendments
on tenders.id_llamado = amendments.id_llamado
group by 1;

-- Are there adendas that are before tender publication date?
-- Tested with fecha_archivo and fecha_publicacion. Same result: Most of the adendas come after tender publication date.
select 
sum(case when (tenders.fecha_publicacion_convocatoria > amendments.fecha_archivo and amendments.tipo_documento = 'Adenda') then 1 else 0 end) Q_Adendas_before_publication_date,
sum(case when (tenders.fecha_publicacion_convocatoria < amendments.fecha_archivo and amendments.tipo_documento = 'Adenda') then 1 else 0 end) Q_Adendas_after_publication_date,
sum(case when (tenders.fecha_publicacion_convocatoria > amendments.fecha_publicacion and amendments.tipo_documento = 'Adenda') then 1 else 0 end) Q_Adendas_before_pubdate_publication_date,
sum(case when (tenders.fecha_publicacion_convocatoria < amendments.fecha_publicacion and amendments.tipo_documento = 'Adenda') then 1 else 0 end) Q_Adendas_after_pubdate_publication_date
from raw.proceso tenders
inner join raw.pbc_adenda amendments
on tenders.id_llamado = amendments.id_llamado;

-- Are there PBC that are after tender publication date?
-- Most of the PBC(Tender documents) are before tender publication date 
select 
sum(case when (tenders.fecha_publicacion_convocatoria > amendments.fecha_archivo and amendments.tipo_documento = 'Pliego de bases y Condiciones') then 1 else 0 end) Q_PBC_before_publication_date,
sum(case when (tenders.fecha_publicacion_convocatoria < amendments.fecha_archivo and amendments.tipo_documento = 'Pliego de bases y Condiciones') then 1 else 0 end) Q_PBC_after_publication_date
from raw.proceso tenders
inner join raw.pbc_adenda amendments
on tenders.id_llamado = amendments.id_llamado;
	
-- How many adendas you have in each part of process? Before and after tender publication date // Before and after complaints
-- You have both: Adenda before complaints and after complaints.
select 
sum(case when (tenders.fecha_publicacion_convocatoria > amendments.fecha_archivo and amendments.tipo_documento = 'Adenda') then 1 else 0 end) Q_Adendas_before_publication_date,
sum(case when (tenders.fecha_publicacion_convocatoria < amendments.fecha_archivo and amendments.tipo_documento = 'Adenda') then 1 else 0 end) Q_Adendas_after_publication_date,
sum(case when (tenders.fecha_publicacion_convocatoria < amendments.fecha_archivo and amendments.tipo_documento = 'Adenda' and amendments.fecha_archivo < complaints.fecha_resolucion_proceso) then 1 else 0 end) Q_Adendas_after_publication_date_before_complaint,
sum(case when (tenders.fecha_publicacion_convocatoria < amendments.fecha_archivo and amendments.tipo_documento = 'Adenda' and amendments.fecha_archivo > complaints.fecha_resolucion_proceso) then 1 else 0 end) Q_Adendas_after_publication_date_after_complaint,
sum(case when (tenders.fecha_publicacion_convocatoria < amendments.fecha_archivo and amendments.tipo_documento = 'Adenda' and complaints.fecha_resolucion_proceso is null) then 1 else 0 end) Q_Adendas_after_publication_no_complaints
from raw.proceso tenders
inner join raw.pbc_adenda amendments
on tenders.id_llamado = amendments.id_llamado
left join raw.proceso_juridico complaints
on tenders.id_llamado = complaints.id_llamado;

------------------ 2.  Complaints

-- Sanity Check: Nulls
select * from raw.proceso_juridico limit 10;
select count(*) from raw.proceso_juridico;
select count(*) from raw.proceso_juridico where id_llamado is null;
select count(*) from raw.proceso_juridico where convocatoria_slug is null;
select count(*) from raw.proceso_juridico where juez_instructor is null;
select count(*) from raw.proceso_juridico where fecha_resolucion_proceso is null;
select count(*) from raw.proceso_juridico where resultado is null;

-- Sanity Check: If all complaints could match to tender IDs: Ok! But More than 1 complain for 1 tender
select
	count(*)
from	
	raw.proceso_juridico complaints
inner join
	raw.proceso tenders
on complaints.id_llamado = tenders.id_llamado;

select 
	count (distinct id_llamado ) 
from 
	raw.proceso_juridico; -- More than 1 complain for the same tender.
	
-- Sanity check: Example of tenders with many complaints
select
	id_llamado, 
	resultado
from	
	raw.proceso_juridico
group by id_llamado, resultado
having count(id_llamado) > 1
order by id_llamado;


-- Complaints results and result detail
select
	_tipo_resultado,
	count(distinct resultado) quantity_distinct_categories,
	count(*) quantity_cases
from
	raw.proceso_juridico
group by  _tipo_resultado;

select 
	_tipo_resultado,
	resultado,
	count(*) quantity
from 
	raw.proceso_juridico
group by _tipo_resultado, resultado
order by _tipo_resultado, quantity desc;

-- Complaints in time
select	
	extract(year from fecha_resolucion_proceso) * 100 + extract (month from fecha_resolucion_proceso) yearmonth,
	sum(case when _tipo_resultado = 'a_favor' then 1 else 0 end) favorable,
	sum(case when _tipo_resultado = 'en_contra' then 1 else 0 end) not_favorable,
	sum(case when _tipo_resultado is null then 1 else 0 end) result_null,
	count(*) total_complaints
from	
	raw.proceso_juridico
group by 1
order by 1 desc;

-- Distribution of judges (11) by effective complaints vs not effective (17)
with judges_data as(
select 
	juez_instructor, 
	sum(case when _tipo_resultado = 'a_favor' then 1 else 0 end) favorable,
	sum(case when _tipo_resultado = 'en_contra' then 1 else 0 end) not_favorable,
	sum(case when _tipo_resultado is null then 1 else 0 end) result_null,
	count(*) total_complaints,
	count (distinct denunciante) distinct_complainant
from 
	raw.proceso_juridico
group by 
	juez_instructor)

select 
	judges_data.total_complaints,
	judges_data.distinct_complainant,
	round( judges_data.favorable * 1.00 / judges_data.total_complaints * 1.00 ,2)  ratio_favorable,
	round(judges_data.not_favorable * 1.00 / judges_data.total_complaints * 1.00,2) ratio_notfavorable,
	round(judges_data.result_null  * 1.00 / judges_data.total_complaints * 1.00,2) ratio_null
from
	judges_data
order by total_complaints desc;

-- Are there companies that repeatedly makes complaints? (10)
with complainants as (
select
	denunciante, 
	sum(case when _tipo_resultado = 'a_favor' then 1 else 0 end) favorable,
	sum(case when _tipo_resultado = 'en_contra' then 1 else 0 end) not_favorable,
	sum(case when _tipo_resultado is null then 1 else 0 end) result_null,
	count(id_llamado) total_complaints,
	count(distinct id_llamado) total_distinct_tenders_complaints
from	
	raw.proceso_juridico
group by 
	denunciante
order by 
	total_complaints desc)
	
select 
	complainants.*,
	round(complainants.favorable * 1.00 / complainants.total_complaints * 1.00 ,2)  ratio_favorable,
	round(complainants.not_favorable * 1.00 / complainants.total_complaints * 1.00,2) ratio_notfavorable,
	round(complainants.result_null  * 1.00 / complainants.total_complaints * 1.00,2) ratio_null
from	
	complainants
order by total_complaints desc;

------------------ 3.  Amendments 2: Adendas after complaints
-- How many adendas effective complaints have?
with complaints_amendments as(
select
	complaints.id_llamado,
	count(amendments.id) quantity_amendments
from 
	raw.proceso_juridico complaints
left join	
	raw.pbc_adenda amendments
on complaints.id_llamado = amendments.id_llamado
and amendments.fecha_archivo > complaints.fecha_resolucion_proceso
where _tipo_resultado = 'a_favor' 
group by 1)

select  
	count(*) total_complaints,
	sum(case when quantity_amendments > 1 then 1 else 0 end) complaints_with_more_than_1_amendment,
	sum(case when quantity_amendments = 0 then 1 else 0 end) complaints_with_no_amendments,
	avg(quantity_amendments) avg_amendments_per_complaint
from complaints_amendments;


-------------- 4. Sumarizing reviewing process, PBC (tender documents), Amendments 1 and amendments 2 at tender level
-- Adding data about complaints, reviewing process and amendments 1 (related to tender) and amendments 2 (related to complaints)
with proceso_juridico_flatten as (
select
	id_llamado,
	count(distinct _tipo_resultado) q_distinct_complaints_results,
	sum(case when _tipo_resultado = 'a_favor' then 1 else 0 end) q_effective_complaints,
	sum(case when _tipo_resultado = 'en_contra' then 1 else 0 end) q_noneffective_complaints,
	min(fecha_resolucion_proceso) min_fecha_resolucion_proceso,
	max(fecha_resolucion_proceso) max_fecha_resolucion_proceso
from	
	raw.proceso_juridico
group by 
	id_llamado)

, data_to_plot as (
select	
	tenders.id_llamado,
	tenders.monto_total_adjudicado,
	tenders.tipo_procedimiento,
	max(review.audit_date)::date - min(review.reception_date)::date Reviewing_time_in_days,
	count(distinct review.id) Reviewing_quantity,
	sum(case when (amendments.tipo_documento = 'Pliego de bases y Condiciones' and tenders.fecha_publicacion_convocatoria > amendments.fecha_archivo) then 1 else 0 end) Q_PBC_before_publication_date,
	sum(case when (amendments.tipo_documento = 'Adenda' and tenders.fecha_publicacion_convocatoria > amendments.fecha_archivo) then 1 else 0 end) Q_Adendas_before_publication_date,
	tenders.fecha_publicacion_convocatoria,
	sum(case when (amendments.tipo_documento = 'Pliego de bases y Condiciones' and tenders.fecha_publicacion_convocatoria < amendments.fecha_archivo and amendments.fecha_archivo < complaints.min_fecha_resolucion_proceso) then 1 else 0 end) Q_PBC_after_publication_date,
	sum(case when (amendments.tipo_documento = 'Adenda' and tenders.fecha_publicacion_convocatoria < amendments.fecha_archivo and amendments.fecha_archivo < complaints.min_fecha_resolucion_proceso) then 1 else 0 end) Q_Adendas_after_publication_date_before_complaint,
	complaints.min_fecha_resolucion_proceso, 
	coalesce(complaints.q_distinct_complaints_results,0) q_distinct_complaints_results,
	complaints.q_effective_complaints,
	complaints.q_noneffective_complaints,
	sum(case when (amendments.tipo_documento = 'Pliego de bases y Condiciones' and tenders.fecha_publicacion_convocatoria < amendments.fecha_archivo and amendments.fecha_archivo > complaints.min_fecha_resolucion_proceso) then 1 else 0 end) Q_PBC_after_publication_date_after_complaint,
	sum(case when (amendments.tipo_documento = 'Adenda' and tenders.fecha_publicacion_convocatoria < amendments.fecha_archivo and amendments.fecha_archivo > complaints.min_fecha_resolucion_proceso) then 1 else 0 end) Q_Adendas_after_publication_date_after_complaint
from
	raw.proceso tenders
left join	
	raw.tender_record review
on tenders.id_llamado = review.llamado_id
left join
	raw.pbc_adenda amendments
on tenders.id_llamado = amendments.id_llamado
left join
	proceso_juridico_flatten complaints
on tenders.id_llamado = complaints.id_llamado
group by 	
	tenders.id_llamado,
	tenders.monto_total_adjudicado,
	tenders.tipo_procedimiento,
	tenders.fecha_publicacion_convocatoria,
	complaints.min_fecha_resolucion_proceso, 
	complaints.q_distinct_complaints_results,
	complaints.q_effective_complaints,
	complaints.q_noneffective_complaints)
	
select * from data_to_plot;





