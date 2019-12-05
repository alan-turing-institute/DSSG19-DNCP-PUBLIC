# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
# Run from terminal
# python features-generator.py run_sql_query entity,short_name,entity_rename
# python features-generator.py run_sql_query 'tipo_procedimiento_codigo','tp','procurement_type'
# -

# Import Scripts
# %reload_ext autoreload
# %autoreload 2
import sys
sys.path.insert(0, '../../src/utils')
import utils

# Packages
import pandas as pd
import fire

# ### Start connection with db
#
#

con = utils.connect_to_database()

# ### Select entities to build features online

# +
# Name of entity/object to build features on (agency,tipo_procedimiento, etc)
entity = 'tipo_procedimiento_codigo'

# Assign a short name for the entity to be use in the variables name
short_name = 'tp'

entity_rename = 'procurement_type'

# To run test: select only one entity
# and add it in where clause in the first table(tmp_{entity}.{entity}_dates)

#entity_example = 'CE'
# -

# ### Start to execute SQL queries
def run_sql_query(entity,short_name,entity_rename):

    # Create temporary schema to put all the tables
    query = f"""
    drop schema if exists tmp_{entity} cascade;
    create schema tmp_{entity};"""
    con.execute(query)

    ###### Create date for each entity
    # tmp_{entity}.{entity}_dates
    # Create date for each entity when an event occured to the entity (new tender,new complaint, bidders)
    query = f"""
    create table tmp_{entity}.{entity}_dates as (
    -- Dates related to new tender
        select
            {entity}
            , reception_date as date
        from
            semantic.tenders

        union

     -- Dates related to bidders finish dates
        select
            {entity}
            , fecha_entrega_oferta as date
        from
            raw.proceso

        union

     -- Dates related to new complaint resolution
        select
            {entity}
            , fecha_resolucion_proceso as date
        from
            raw.proceso tenders
        inner join
            raw.proceso_juridico complaints
        on tenders.id_llamado = complaints.id_llamado);

    -- Add indexes
    drop index if exists {entity}_dates_{entity}_idx ;
    create index {entity}_dates_{entity}_idx ON tmp_{entity}.{entity}_dates ({entity});
    drop index if exists {entity}_dates_date_idx ;
    create index {entity}_dates_date_idx ON tmp_{entity}.{entity}_dates (date);
    """
    con.execute(query)

    # +
    # Link each id llamado with a complaint and resolution date.
    # tmp_{entity}.tender_{entity}_label
    #Link each id_llamado with a complaint result and a date of resolution, even if there were no complaints.
    #This is done to avoid dealing with Null values afterwards.
    # - For processes without complaints, we set the resolution date to an old date, so that we cound them all.
    # - Resolution results will be:
    #    a_favor
    #    en_contra
    #    cancelled
    #    no_complaints

    query = f"""
    create table tmp_{entity}.tender_{entity}_label as (
        select
        labels.*
        , case
            when complaints.fecha_resolucion_proceso is null then '1900-01-01'
            else complaints.fecha_resolucion_proceso end as compl_resolution_date
        , case
          when complaints."_tipo_resultado" is null then
            case when complaints.fecha_resolucion_proceso is null then 'no_complaints'
            else 'cancelled'
            end
          else complaints."_tipo_resultado"
          end
        as result
            , tenders.{entity} as {entity}
            , tenders.amt_planned_real_guarani as amt_planned_real_guarani
            , tenders.log_amt_planned_real_guarani as log_amt_planned_real_guarani
        from semantic.labels as labels
        left join semantic.tenders as tenders
        on labels.id_llamado = tenders.id_llamado
        left join semantic.complaints as complaints
        on labels.id_llamado = complaints.id_llamado
    );


    drop index if exists tender_{entity}_label_{entity}_idx;
    create index tender_{entity}_label_{entity}_idx ON tmp_{entity}.tender_{entity}_label ({entity});
    drop index if exists tender_{entity}_label_publ_date_idx;
    create index tender_{entity}_label_publ_date_idx ON tmp_{entity}.tender_{entity}_label (compl_resolution_date);
    """
    con.execute(query)

    # +
    #/* tmp_{entity}.{entity}_info
    # For each {entity} and date, calculate related information. Mainly:
    # - Number of total tenders received up to date d
    # - Average planned value of the tenders

    query = f"""
    create table tmp_{entity}.{entity}_info as (
        select
            tmp_{entity}.{entity}_dates.*
          , aggr.*
        from tmp_{entity}.{entity}_dates
        left join lateral (
            select count(distinct id_llamado) as n_tenders  -- Number of tenders received
                , avg(distinct log_amt_planned_real_guarani) as avg_log_planned_real_value  -- Average planned value of tenders in log scale.
                , percentile_cont(0.5) within group(order by log_amt_planned_real_guarani::float) as median_log_planned_real_value -- Median planned value of tenders in log scale.
                , percentile_cont(0.25) within group(order by log_amt_planned_real_guarani::float) as quarter_percentil_log_planned_real_value -- Percentil planned value of tenders in log sacale.
                , percentile_cont(0.75) within group(order by log_amt_planned_real_guarani::float) as seventy_five_percentil_log_planned_real_value -- Percentil planned value of tenders in log sacale.
                , percentile_cont(0.02) within group(order by log_amt_planned_real_guarani::float) as two_percentil_log_planned_real_value -- Percentil planned value of tenders in log sacale.
                , percentile_cont(0.98) within group(order by log_amt_planned_real_guarani::float) as ninety_eight_percentil_log_planned_real_value -- Percentil planned value of tenders in log sacale.
            from tmp_{entity}.tender_{entity}_label
            where {entity} = tmp_{entity}.{entity}_dates.{entity} and date(reception_date) <= tmp_{entity}.{entity}_dates.date
            group by {entity}) aggr
        on true
    );

    drop index if exists {entity}_info_{entity}_idx;
    create index {entity}_info_{entity}_idx ON tmp_{entity}.{entity}_info ({entity});
    """
    con.execute(query)

    # +
    # tmp_{entity}.{entity}_complaints_info
    # For each {entity} and date, calculate the number of tenders with effective complaints,
    # as well as the total number of effective complaints, before that date
    # Note that the date used to consider complaints is the resolution date of the complaint process.

    query = f"""
    create table tmp_{entity}.{entity}_complaints_info as (
        select
            tmp_{entity}.{entity}_dates.*
            , aggr.*
        from tmp_{entity}.{entity}_dates
        left join lateral (
            select
                sum(case when result in ('a_favor', 'en_contra') then 1 else 0 end) as n_complaints  -- Number of complaints. If None, set to 0.
                , sum(case when result = 'a_favor' then 1 else 0 end) as n_eff_complaints  -- N. of effective complaints. If None, set to 0.
                , sum(case when result = 'cancelled' then 1 else 0 end) as n_cancelled_complaints  -- N. of cancelled complaints. If None, set to 0.
                , count(distinct id_llamado) filter (where result in ('a_favor', 'en_contra')) as n_tenders_complaints  -- N. of unique tenders with complaints. If None, set to 0.
                , count(distinct id_llamado) filter (where result = 'a_favor') as n_tenders_eff_complaints  -- N. of unique tenders with eff. complaints. If None, set to 0.
            from tmp_{entity}.tender_{entity}_label as a
            where {entity} = tmp_{entity}.{entity}_dates.{entity} and compl_resolution_date < tmp_{entity}.{entity}_dates.date
            group by {entity}) aggr
        on true
    );"""

    con.execute(query)
    # -

    # #### Bidders

    # +
    # Table that counts quantity of bidders
    # that participated in each tender
    # and closing date to present bid offer

    query = f"""
    drop table if exists tmp_{entity}.bidders;
    create table tmp_{entity}.bidders as (
    select
        tenders.id_llamado
        , tenders.{entity} as {entity}
        , tenders.fecha_entrega_oferta bidding_recepetion_date
        , count(distinct bidder_unique_identifier) quantity_bidders
    from
        raw.proceso tenders
    inner join
        cleaned.oferentes bidders
    on tenders.id_llamado = bidders.id_llamado
    group by 1,2,3
    );


    -- Aggregates of bidders for each {entity} previuos
    -- to each date
    -- drop table tmp_{entity}.{entity}_bidders;

    create table tmp_{entity}.{entity}_bidders as (
        select
            tmp_{entity}.{entity}_dates.*
            , aggr.*
        from tmp_{entity}.{entity}_dates
        left join lateral (
            select
                sum(quantity_bidders) total_n_bidders
                , avg(quantity_bidders) avg_n_bidders
                , percentile_cont(0.5) within group(order by quantity_bidders) as median_n_bidders
            from tmp_{entity}.bidders as a
            where {entity} = tmp_{entity}.{entity}_dates.{entity} and bidding_recepetion_date < tmp_{entity}.{entity}_dates.date
            group by {entity}) aggr
        on true
    );

    -- Add a variable that identifies if bidder data is missing
    create table tmp_{entity}.bidders_missings as (
    select
        *
        , case when total_n_bidders is null then 1 else 0 end as missing_bidders_data
    from
        tmp_{entity}.{entity}_bidders);

    -- Identify the oldest data we have about a variable and assigned to
    -- all missing values

    -- Identify value of the first data we have
    drop table if exists tmp_{entity}.min_n_bidders;
    create table tmp_{entity}.min_n_bidders as (
    select
        tmp_{entity}.bidders_missings.{entity}
        , total_n_bidders as min_total_n_bidders
        , avg_n_bidders as min_avg_n_bidders
        , median_n_bidders as min_median_n_bidders
    from
        tmp_{entity}.bidders_missings
    left join
        (select
        {entity},
        min (date) minimum_date
        from tmp_{entity}.bidders_missings
        where missing_bidders_data = 0
        group by 1) a
    on tmp_{entity}.bidders_missings.{entity} = a.{entity}
    where tmp_{entity}.bidders_missings.date = a.minimum_date);

    -- Compute missing values
    drop table if exists tmp_{entity}.{entity}_bidders_completed_cases;
    create table tmp_{entity}.{entity}_bidders_completed_cases as (
    select
        tmp_{entity}.bidders_missings.{entity}
        , date
        , case when (tmp_{entity}.bidders_missings.{entity} = tmp_{entity}.min_n_bidders.{entity} and total_n_bidders is null) then min_total_n_bidders else  total_n_bidders end total_n_bidders
        , case when (tmp_{entity}.bidders_missings.{entity} = tmp_{entity}.min_n_bidders.{entity} and avg_n_bidders is null) then min_total_n_bidders else  avg_n_bidders end avg_n_bidders
        , case when (tmp_{entity}.bidders_missings.{entity} = tmp_{entity}.min_n_bidders.{entity} and median_n_bidders is null) then min_total_n_bidders else  median_n_bidders end median_n_bidders
        , missing_bidders_data
    from tmp_{entity}.bidders_missings
    left join tmp_{entity}.min_n_bidders
    on tmp_{entity}.bidders_missings.{entity} = tmp_{entity}.min_n_bidders.{entity});

    """

    con.execute(query)
    # -

    # ### Products

    # unique_number_products
    # mean_number_products
    # median_number_products
    # p25_number_products
    # p75_number_products
    # high_risk_bool
    # mode_presentacion
    query = f"""
    drop table if exists tmp_{entity}.product_distinct_counts;
    create table tmp_{entity}.product_distinct_counts as (
    select
    process.id_llamado, process.reception_date, process.{entity}, product.distinct_counts_products
    from semantic.tenders as process
        left join (
            select id_llamado, count(distinct producto_nombre_catalogo) distinct_counts_products
            from cleaned.items as item
            group by item.id_llamado
    ) as product
    on product.id_llamado = process.id_llamado
    );

    --Whether the {entity} submitted a tender of the top 10 high risk product category in the past 2 years

    --Get top 10 high risk product category

    create table tmp_{entity}.top_10_high_risk as (
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
    create table tmp_{entity}.high_risk_tender as (
        with high_risk as
            (select avg(i.id_llamado) id_llamado, bool_or(i.producto_nombre_catalogo in (
                select producto_nombre_catalogo
                from tmp_{entity}.top_10_high_risk))
                as high_risk_bool
            from cleaned.items i
            group by id_llamado)
        select high_risk.id_llamado, high_risk_bool, tender.reception_date
        from high_risk
        left join semantic.tenders as tender
        on high_risk.id_llamado=tender.id_llamado
        where reception_date is not null);

    --Putting the two tender level tables together

    create table tmp_{entity}.product_tender as (
        select p.*, r.high_risk_bool
        from tmp_{entity}.high_risk_tender r
            left join tmp_{entity}.product_distinct_counts p
            on r.id_llamado = p.id_llamado and r.reception_date = p.reception_date
        order by high_risk_bool desc
    );

    --create a lateral join to join product counts features with dates on {entity} level
    create table tmp_{entity}.product_counts_joined as(
        select
            tmp_{entity}.{entity}_dates.{entity},
            tmp_{entity}.{entity}_dates.date,
            aggr.*
        from tmp_{entity}.{entity}_dates
        left join lateral (
            select
                sum(distinct_counts_products) unique_number_products
                , avg(distinct_counts_products) mean_number_products
                , percentile_cont(0.5) within group(order by distinct_counts_products) as median_number_products
                , percentile_cont(0.25) within group(order by distinct_counts_products) as p25_number_products
                , percentile_cont(0.75) within group(order by distinct_counts_products) as p75_number_products
                , bool_or(high_risk_bool) as high_risk_bool
            from tmp_{entity}.product_tender as a
            where a.{entity} = tmp_{entity}.{entity}_dates.{entity} and a.reception_date < tmp_{entity}.{entity}_dates.date
            group by {entity}) aggr
        on true
    );
    """
    con.execute(query)

    # #### Final tables

    # Create the final {entity} features table, adding the number of effective complaints in the last
    #  6 months, 1 year, etc.
    query = f"""
    drop table if exists semantic.{entity};
    create table semantic.{entity} as
    select *
        , n_complaints - lag(n_complaints, 90) over w as n_complaints_3m -- Quantity of complaints in the last 6 month. If null, not enough time has passed by to calculate it.
        , n_eff_complaints - lag(n_eff_complaints, 90) over w as n_eff_complaints_3m -- Quantity of effective complaints in the last 6 month. If null, not enough time has passed by to calculate it.
        , n_tenders_complaints - lag(n_tenders_complaints, 90) over w as n_tenders_complaints_3m -- Quantity of tenders that were complained in the last 6 month. If null, not enough time has passed by to calculate it.
        , n_tenders_eff_complaints - lag(n_tenders_eff_complaints, 90) over w as n_tenders_eff_complaints_3m -- Quantity of tenders with effective complaints in the last 6 month. If null, not enough time has passed by to calculate it.
    --  , n_complaints - lag(n_complaints, 365) over w as n_complaints_1y -- Quantity of complaints in the previous 365 days. If null, not enough time has passed by to calculate it.
    --  , n_eff_complaints - lag(n_eff_complaints, 365) over w as n_eff_complaints_1y -- Quantity of complaints in the previous 365 days. If null, not enough time has passed by to calculate it.
    --  , n_tenders_complaints - lag(n_tenders_complaints, 365) over w as n_tenders_complaints_1y -- Quantity of complaints in the previous 365 days. If null, not enough time has passed by to calculate it.
    --  , n_tenders_eff_complaints - lag(n_tenders_eff_complaints, 365) over w as n_tenders_eff_complaints_1y -- Quantity of complaints in the previous 365 days. If null, not enough time has passed by to calculate it.
    from tmp_{entity}.{entity}_complaints_info
    left join tmp_{entity}.{entity}_info
    using ({entity}, date)
    left join tmp_{entity}.{entity}_bidders_completed_cases bidders
    using ({entity}, date)
    left join tmp_{entity}.product_counts_joined
    using ({entity}, date)
    --left join tmp_{entity}.mode_packaging
    --using ({entity}, date)
    window w as (
      partition by tmp_{entity}.{entity}_complaints_info.{entity}
      order by tmp_{entity}.{entity}_complaints_info.date
    );

    -- Ensure that there are no nulls
    delete from semantic.{entity} where not ({entity} is not null);

    -- Add indexes
    -- drop index if exists {entity}_{entity}_idx;
    -- create index {entity}_{entity}_idx ON semantic.{entity} ({entity});
    -- drop index if exists {entity}_date_idx;
    -- create index {entity}_date_idx ON semantic.{entity}(date);

    -- Create table linking tenders with {entity}' features

    drop table if exists semantic.tenders_{entity};
    create table semantic.tenders_{entity} as
    select
        tenders.id_llamado
        , {entity}.*
    from semantic.tenders as tenders
    left join semantic.{entity} as {entity}
    on tenders.{entity} = {entity}.{entity} and tenders.reception_date = {entity}.date;
    """
    con.execute(query)

    ## Delete nulls
    query = f"""
    delete from semantic.tenders_{entity}
    where not ({entity} is not null)
    """
    con.execute(query)

    ## Rename columns

    query = f"""
    select * from semantic.tenders_{entity}
    """
    df = pd.read_sql_query(query, con)

    columns = list(df.columns)

    for column in columns:
        if column == 'id_llamado' or column =='date' or column == entity:
            pass
        else:
            old_column = column
            new_column = short_name + '_' + column
            query = f"""
            alter table semantic.tenders_{entity} rename column {old_column} to {new_column}
            """
            con.execute(query)

    ## Rename tables
    query = f"""
    drop table if exists semantic.tenders_{entity_rename};
    alter table semantic.tenders_{entity}
    rename to tenders_{entity_rename};
    """
    con.execute(query)

    query = f"""
    drop table if exists semantic.{entity_rename};
    alter table semantic.{entity}
    rename to {entity_rename};
    """
    con.execute(query)

    ### Drop all temp tables in temporary schema

    query = f"""drop schema tmp_{entity} cascade;"""
    con.execute(query)



# +
# run_sql_query(entity = entity, short_name = short_name, entity_rename = entity_rename)
# -

if __name__== '__main__':
    fire.Fire()
