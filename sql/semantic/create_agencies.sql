/*
 Creates agencies table that is indexed by agency and by date.
 We create temporal tables to help create the final features agencies table.
 For each agency and date, we have a snapshot of the situation then.
*/
--create temporary schema
drop schema if exists tmp cascade;
create schema tmp;

-- tmp.agency_dates
-- Create a series of dates for each agency: from the oldest tender to the most recent one
create table tmp.agency_dates as (
    select agency, municipality_bool, date(generate_series(min_date, max_date, '1 day'::interval)) as date
	from (
		select
		convocante as agency
		, min(date_trunc('day', reception_date)) as min_date
		, max(date_trunc('day', reception_date)) as max_date
		, case when tipo_entidad = 'Municipalidades' then 1 else 0 end as municipality_bool
		from semantic.tenders
		--where convocante in( 'Ministerio de la Niñez y la Adolescencia (MINNA)','Ministerio de la Mujer')
		group by agency, municipality_bool
	) as min_max_agency
);


-- Add indexes
create index agency_dates_agency_idx ON tmp.agency_dates (agency);
create index agency_dates_date_idx ON tmp.agency_dates (date);

/* tmp.tender_agency_label
Link each id_llamado with a complaint result and a date of resolution, even if there were no complaints.
This is done to avoid dealing with Null values afterwards.
- For processes without complaints, we set the resolution date to an old date, so that we cound them all.
- Resolution results will be:
    a_favor
    en_contra
    cancelled
    no_complaints
*/

create table tmp.tender_agency_label as (
	select labels.*
		, case
			when complaints.fecha_resolucion_proceso is null then '1900-01-01'
			else complaints.fecha_resolucion_proceso end
		as compl_resolution_date
		, case
			when complaints."_tipo_resultado" is null then
				case when complaints.fecha_resolucion_proceso is null then 'no_complaints'
				else 'cancelled'
				end
			else complaints."_tipo_resultado"
			end
		as result
		, tenders.convocante as agency
		, tenders.amt_planned_real_guarani
		, tenders.log_amt_planned_real_guarani
	from semantic.labels as labels
	left join semantic.tenders as tenders
	on labels.id_llamado = tenders.id_llamado
	left join semantic.complaints as complaints
	on labels.id_llamado = complaints.id_llamado
);

-- Add indexes
create index tender_agency_label_agency_idx ON tmp.tender_agency_label (agency);
create index tender_agency_label_publ_date_idx ON tmp.tender_agency_label (compl_resolution_date);


/* tmp.agency_info
For each agency and date, calculate related information. Mainly:
- Number of total tenders received up to date d
- Average planned value of the tenders
*/

create table tmp.agency_info as (
	select tmp.agency_dates.*
		  , aggr.*
	from tmp.agency_dates
	left join lateral (
		select count(distinct id_llamado) as n_tenders  -- Number of tenders received
			, avg(distinct log_amt_planned_real_guarani) as avg_log_planned_real_value  -- Average planned value of tenders in log scale.
            , percentile_cont(0.5) within group(order by log_amt_planned_real_guarani::float) as median_log_planned_real_value -- Median planned value of tenders in log scale.
			, percentile_cont(0.25) within group(order by log_amt_planned_real_guarani::float) as quarter_percentil_log_planned_real_value -- Percentil planned value of tenders in log sacale.
			, percentile_cont(0.75) within group(order by log_amt_planned_real_guarani::float) as seventy_five_percentil_log_planned_real_value -- Percentil planned value of tenders in log sacale.
			, percentile_cont(0.02) within group(order by log_amt_planned_real_guarani::float) as two_percentil_log_planned_real_value -- Percentil planned value of tenders in log sacale.
			, percentile_cont(0.98) within group(order by log_amt_planned_real_guarani::float) as ninety_eight_percentil_log_planned_real_value -- Percentil planned value of tenders in log sacale.
        from tmp.tender_agency_label
		where agency = tmp.agency_dates.agency and date(reception_date) <= tmp.agency_dates.date
		group by agency) aggr
	on true
);

-- Add indexes
create index agency_info_agency_idx ON tmp.agency_info (agency);

/* Kurtosis and Skewness
 * It can be calculated used agg functions but we need to make it work with lateral joins
 * To makefunction stats_agg available it is needed to run
 *'sql/PostgreSQL-Stats_Aggregation.sql'
 */

with kurtosis_skewness as (
	select unnest(array[log_amt_planned_real_guarani::float]) n from tmp.tender_agency_label
)
select
  (stats_agg(n)).kurtosis kurtosis
, (stats_agg(n)).skewness skewness
from kurtosis_skewness;


/* tmp.agencies_complaints_info
For each agency and date, calculate the number of tenders with effective complaints,
as well as the total number of effective complaints, before that date
Note that the date used to consider complaints is the resolution date of the complaint process.
*/

create table tmp.agencies_complaints_info as (
	select tmp.agency_dates.*
		, aggr.*
	from tmp.agency_dates
	left join lateral (
		select
            sum(case when result in ('a_favor', 'en_contra') then 1 else 0 end) as n_complaints  -- Number of complaints. If None, set to 0.
			, sum(case when result = 'a_favor' then 1 else 0 end) as n_eff_complaints  -- N. of effective complaints. If None, set to 0.
            , sum(case when result = 'cancelled' then 1 else 0 end) as n_cancelled_complaints  -- N. of cancelled complaints. If None, set to 0.
			, count(distinct id_llamado) filter (where result in ('a_favor', 'en_contra')) as n_tenders_complaints  -- N. of unique tenders with complaints. If None, set to 0.
			, count(distinct id_llamado) filter (where result = 'a_favor') as n_tenders_eff_complaints  -- N. of unique tenders with eff. complaints. If None, set to 0.
		from tmp.tender_agency_label as a
		where agency = tmp.agency_dates.agency and compl_resolution_date < tmp.agency_dates.date
		group by agency) aggr
	on true
);

-----------------
---- BIDDERS ----
-----------------
-- Table that counts quantity of bidders
-- that participated in each tender
-- and closing date to present bid offer


create table tmp.bidders as (
select
	tenders.id_llamado
	, tenders.convocante as agency
	, tenders.fecha_entrega_oferta bidding_recepetion_date
	, count(distinct bidder_unique_identifier) quantity_bidders
from
	raw.proceso tenders
inner join
	cleaned.oferentes bidders
on tenders.id_llamado = bidders.id_llamado
group by 1,2,3
);

-- Aggregates of bidders for each agency previuos
-- to each date 
-- drop table tmp.agencies_bidders;

create table tmp.agencies_bidders as (
	select
		tmp.agency_dates.*
		, aggr.*
	from tmp.agency_dates
	left join lateral (
		select
			sum(quantity_bidders) total_n_bidders
			, avg(quantity_bidders) avg_n_bidders
			, percentile_cont(0.5) within group(order by quantity_bidders) as median_n_bidders
		from tmp.bidders as a
		where agency = tmp.agency_dates.agency and bidding_recepetion_date < tmp.agency_dates.date
		group by agency) aggr
	on true
);

/*Add a variable that identifies if bidder data
 * is missing
*/ 

create table tmp.bidders_missings as (
select 
	*
	, case when total_n_bidders is null then 1 else 0 end as missing_bidders_data
from 
	tmp.agencies_bidders);

/* Identify the most antique data we have 
 * about a variable and assigned to
 * all missing values
 */

-- Identify value of the first data we have
--drop table tmp.min_n_bidders;

create table tmp.min_n_bidders as (
select 
	tmp.bidders_missings.agency
	, total_n_bidders as min_total_n_bidders
	, avg_n_bidders as min_avg_n_bidders
	, median_n_bidders as min_median_n_bidders
from 
	tmp.bidders_missings 
left join 
	(select 
	agency,
	min (date) minimum_date
	from tmp.bidders_missings
	where missing_bidders_data = 0
	group by 1) a 
on tmp.bidders_missings.agency = a.agency
where tmp.bidders_missings.date = a.minimum_date);

-- Compute missing values
-- drop table tmp.agencies_bidders_completed_cases;

create table tmp.agencies_bidders_completed_cases as (
select 
tmp.bidders_missings.agency
, municipality_bool
, date
, case when (tmp.bidders_missings.agency = tmp.min_n_bidders.agency and total_n_bidders is null) then min_total_n_bidders else  total_n_bidders end total_n_bidders
, case when (tmp.bidders_missings.agency = tmp.min_n_bidders.agency and avg_n_bidders is null) then min_total_n_bidders else  avg_n_bidders end avg_n_bidders
, case when (tmp.bidders_missings.agency = tmp.min_n_bidders.agency and median_n_bidders is null) then min_total_n_bidders else  median_n_bidders end median_n_bidders
, missing_bidders_data
from tmp.bidders_missings
left join tmp.min_n_bidders
on tmp.bidders_missings.agency = tmp.min_n_bidders.agency);


/* Products table
 *  -unique_number_products
	-mean_number_products
	-median_number_products
	-p25_number_products
	-p75_number_products
	-high_risk_bool
	-mode_presentacion
 */

--create a temp table to store tender level results linked with products

create table tmp.product_distinct_counts as (
select 
process.id_llamado, process.reception_date, process.convocante, product.distinct_counts_products
from semantic.tenders as process
	left join (
		select id_llamado, count(distinct producto_nombre_catalogo) distinct_counts_products
		from cleaned.items as item
		group by item.id_llamado
	) as product
on product.id_llamado = process.id_llamado
);

--Whether the agency submitted a tender of the top 10 high risk product category in the past 2 years

	--Get top 10 high risk product category

create table tmp.top_10_high_risk as (
	select j.producto_nombre_catalogo, sum(j.bool_of_effective_complaints) as complaint_counts
	from (
		select i.id_llamado, i.producto_nombre_catalogo, l.bool_of_effective_complaints
		from cleaned.items i
		inner join semantic.labels l
		on l.id_llamado=i.id_llamado) as j
	group by j.producto_nombre_catalogo
	order by complaint_counts desc
	limit 10
);

	--Check if tender has high risk product
create table tmp.high_risk_tender as (
	with high_risk as  
		(select avg(i.id_llamado) id_llamado, bool_or(i.producto_nombre_catalogo in (	
			select producto_nombre_catalogo
			from tmp.top_10_high_risk))
			as high_risk_bool 
		from cleaned.items i
		group by id_llamado)
	select high_risk.id_llamado, high_risk_bool, tender.reception_date
	from high_risk
	left join semantic.tenders as tender
	on high_risk.id_llamado=tender.id_llamado
	where reception_date is not null);

--There are 1k out of 125k id_llamados unmatched between items and tenders table
/*
with item_unique as(
	select item.id_llamado, count(*)
	from cleaned.items item
	group by item.id_llamado
) select count(item_unique.id_llamado), count(tender.id_llamado), count(*)
from item_unique
full join semantic.tenders as tender
on item_unique.id_llamado = tender.id_llamado; 
*/

--Putting the two tender level tables together

create table tmp.product_tender as (
	select p.*, r.high_risk_bool
	from tmp.high_risk_tender r
		left join tmp.product_distinct_counts p
		on r.id_llamado = p.id_llamado and r.reception_date = p.reception_date
	order by high_risk_bool desc
);

--create a lateral join to join product counts features with dates on agency level
create table tmp.product_counts_joined as(
	select
		tmp.agency_dates.agency,
		tmp.agency_dates.date,
		aggr.*
	from tmp.agency_dates
	left join lateral (
		select
			sum(distinct_counts_products) unique_number_products
			, avg(distinct_counts_products) mean_number_products
			, percentile_cont(0.5) within group(order by distinct_counts_products) as median_number_products
			, percentile_cont(0.25) within group(order by distinct_counts_products) as p25_number_products
			, percentile_cont(0.75) within group(order by distinct_counts_products) as p75_number_products
			, bool_or(high_risk_bool) as high_risk_bool
		from tmp.product_tender as a
		where a.convocante = tmp.agency_dates.agency and a.reception_date < tmp.agency_dates.date
		group by agency) aggr
	on true
);

--create lateral joins to get the mode of packaging
--create temp table tmp.mode_packaging as (
--	select tmp.agency_dates.agency,
--			tmp.agency_dates.date,
--			aggr.*
--	from tmp.agency_dates
--	left join lateral (
--		select mode() within group(order by presentacion) as mode_presentacion 
--		from (
--			select item.id_llamado, presentacion, tender.reception_date, tender.convocante
--			from cleaned.items as item
--			inner join semantic.tenders as tender
--			on tender.id_llamado = item.id_llamado
--		) as joined_item
--		where joined_item.convocante = tmp.agency_dates.agency and joined_item.reception_date < tmp.agency_dates.date
--		group by agency) aggr
--		on true);


-- Create the final agencies features table, adding the number of effective complaints in the last
-- 6 months, 1 year, etc.
drop table if exists semantic.agencies;
create table semantic.agencies as
select *
	, n_complaints - lag(n_complaints, 90) over w as n_complaints_3m -- Quantity of complaints in the last 6 month. If null, not enough time has passed by to calculate it.
	, n_eff_complaints - lag(n_eff_complaints, 90) over w as n_eff_complaints_3m -- Quantity of effective complaints in the last 6 month. If null, not enough time has passed by to calculate it.
	, n_tenders_complaints - lag(n_tenders_complaints, 90) over w as n_tenders_complaints_3m -- Quantity of tenders that were complained in the last 6 month. If null, not enough time has passed by to calculate it.
	, n_tenders_eff_complaints - lag(n_tenders_eff_complaints, 90) over w as n_tenders_eff_complaints_3m -- Quantity of tenders with effective complaints in the last 6 month. If null, not enough time has passed by to calculate it.
--	, n_complaints - lag(n_complaints, 365) over w as n_complaints_1y -- Quantity of complaints in the previous 365 days. If null, not enough time has passed by to calculate it.
--	, n_eff_complain	ts - lag(n_eff_complaints, 365) over w as n_eff_complaints_1y -- Quantity of complaints in the previous 365 days. If null, not enough time has passed by to calculate it.
--	, n_tenders_complaints - lag(n_tenders_complaints, 365) over w as n_tenders_complaints_1y -- Quantity of complaints in the previous 365 days. If null, not enough time has passed by to calculate it.
--	, n_tenders_eff_complaints - lag(n_tenders_eff_complaints, 365) over w as n_tenders_eff_complaints_1y -- Quantity of complaints in the previous 365 days. If null, not enough time has passed by to calculate it.
from tmp.agencies_complaints_info
left join tmp.agency_info
using (agency, date, municipality_bool)
left join tmp.agencies_bidders_completed_cases bidders
using (agency, date, municipality_bool)
left join tmp.product_counts_joined
using (agency, date)
--left join tmp.mode_packaging
--using (agency, date)
window w as (
  partition by tmp.agencies_complaints_info.agency
  order by tmp.agencies_complaints_info.date
);

--Ensure that there are no nulls
delete from semantic.agencies where not (agencies is not null);

-- Add indexes
create index agencies_agency_idx ON semantic.agencies (agency);
create index agencies_date_idx ON semantic.agencies (date);

-- Create table linking tenders with agencies' features
-- drop table if exists semantic.tenders_agencies;
drop table if exists semantic.tenders_agencies;
create table semantic.tenders_agencies as
select
    tenders.id_llamado
    , agencies.*
from semantic.tenders as tenders
left join semantic.agencies as agencies
on tenders.convocante = agencies.agency and date_trunc('day', tenders.reception_date) = agencies.date;

drop schema tmp cascade;

