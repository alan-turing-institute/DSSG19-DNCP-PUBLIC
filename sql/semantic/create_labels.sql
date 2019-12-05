/*
 Creates label table that is indexed by id_llamado.
 The initial lables are the number and the existence of effective complaints.
 But, more labels can be added depending on the scope of the project.
*/

drop table if exists semantic.labels;
create table semantic.labels as
select 
	tender.id_llamado
	, min(tender.reception_date) reception_date
	, sum(case when complaint."_tipo_resultado" = 'a_favor' then 1 else 0 end) number_of_effective_complaints -- number of effective complaints
	, max(case when complaint."_tipo_resultado" = 'a_favor' then 1 else 0 end) bool_of_effective_complaints
from
	cleaned.tender_records_cleaned as tender
left join
	raw.proceso_juridico as complaint
on tender.id_llamado = complaint.id_llamado
group by tender.id_llamado;
