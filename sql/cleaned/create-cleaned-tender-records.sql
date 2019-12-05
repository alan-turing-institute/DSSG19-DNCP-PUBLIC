drop table if exists cleaned.tender_records_cleaned;

--Get cleaned variables that exists from the beginning
create table cleaned.tender_records_cleaned as (
select id_llamado,
reception_date
from raw.tender_record
where reception_date is not null
);

