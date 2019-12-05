--Creates the cleaned items table 

drop table if exists cleaned.items;

create table cleaned.items as (
select id_llamado,
lote_descripcion,
producto_nombre_catalogo,
precio_unitario_estimado,
presentacion
from raw.item_solicitado
);

--clean presentacion variable to group _NO_APLICA_ and NULL values together as _NO_APLICA_
update cleaned.items
	set presentacion = '_NO_APLICA_'
	where presentacion = '_NO_APLICA_' or presentacion is null;


