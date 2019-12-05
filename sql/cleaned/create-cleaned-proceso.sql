/* 
 * This file subsets variables from raw schema based on whether the variables are available to
 * tenders when we first received the tenders.
 * The script then looks at variables to see if there are any duplicates in information.
 * Finally it produces a cleaned table for semantic schema
 */ 


drop schema if exists cleaned;
create schema cleaned;

--Create a table that contains the initial variables when a new tender arrives
--Cleans these initial variables
--These variables are found by filtering DNCP's labels of whether they have these variables 
	--at the beginning and also they will not change over time
drop table if exists cleaned.proceso_initial;
create table cleaned.proceso_initial as (
select id_llamado,
planificacion_slug,
convocatoria_slug,
nro_licitacion,
convocante_slug,
convocante_codigo,
convocante,
unidad_contratacion,
tipo_entidad,
categoria_id,
categoria_codigo,
categoria,
_objeto_licitacion,
fecha_publicacion_planificacion,
_moneda,
moneda,
planificacion_url,
tipo_procedimiento, 
tipo_procedimiento_codigo,
monto_global_estimado_planificacion
from raw.proceso);
--note that tipo_procedimiento and tipo_procedimiento_codigo will change minorly so included
--note that currently value is classified as being able to change. we think it will change infrequently.


-----------------------------------------------------
--We want to remove duplicate variables which have the same information
--Before deleting, we check that encodings (codigo) are just shortform of origin variable
with convo_unique as( 
	select distinct convocante_codigo, convocante
from cleaned.proceso_initial
)
select count(*), count(distinct convocante), count(distinct convocante_codigo)
from convo_unique;
--encoding maps exactly

with tipo_unique as( 
	select distinct tipo_procedimiento, tipo_procedimiento_codigo
from cleaned.proceso_initial
)
select count(*), count(distinct tipo_procedimiento), count(distinct tipo_procedimiento_codigo)
from tipo_unique;
--tipo_procedimiento_codigo only has 15 classifications while tipo_procedimiento has 34 classifications
--*Decision to be taken which classification to use
	select tipo_procedimiento_codigo,count(tipo_procedimiento_codigo) as counter
	from cleaned.proceso_initial
	group by tipo_procedimiento_codigo
	order by counter desc;

with category_unique as( 
	select distinct categoria_codigo, categoria, categoria_id
from cleaned.proceso_initial
)
select count(*), count(distinct categoria), count(distinct categoria_codigo), count(distinct categoria_id)
from category_unique;
--categories match exactly

with slug_unique as( 
	select distinct planificacion_slug, convocatoria_slug
from cleaned.proceso_initial
)
select count(*), count(distinct planificacion_slug), count(distinct convocatoria_slug)
from slug_unique;
--some differences in categorisation
select * from cleaned.proceso_initial
where planificacion_slug != convocatoria_slug;
--there are some differences between planificacion_slug and convocatoria_slug
--minor differences sometimes due to -1 or -2 labelled 

with currency_unique as( 
	select distinct moneda, "_moneda"
from cleaned.proceso_initial
)
select count(*), count(distinct moneda), count(distinct "_moneda")
from currency_unique;
--matched exactly 

----------------------------------------------
--cleaned tenders table
drop if exists cleaned.proceso_initial_cleaned;
create table cleaned.proceso_initial_cleaned as (
select id_llamado,
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
--code for taking into account inflation and exchange rate
round(cast(monto_global_estimado_planificacion/(
			select inflation_index/100 from support_docs.inflation
			where id_year = extract(year from tender.fecha_publicacion_planificacion))*(
				select case when tender."_moneda" = 'USD'
				then exchange_usd_guarani
				else 1
				end
				from support_docs.exchange_rate
				where id_year = extract(year from tender.fecha_publicacion_planificacion))
				as numeric),2) amt_planned_real_guarani
from cleaned.proceso_initial as tender);

----------------------------------------------------
--Check NULL values for the important variables
select count(*)- count(id_llamado),
count(*)-count(planificacion_slug),
count(*)-count(convocatoria_slug),
count(*)-count(convocante),
count(*)-count(unidad_contratacion),
count(*)-count(tipo_entidad),
count(*)-count(categoria),
count(*)-count(_objeto_licitacion),
count(*)-count(fecha_publicacion_planificacion),
count(*)-count(_moneda),
count(*)-count(planificacion_url),
count(*)-count(tipo_procedimiento), 
count(*)-count(tipo_procedimiento_codigo),
count(*)-count(amt_planned_real_guarani)
from cleaned.proceso_initial_cleaned;
--no null values for important variables




