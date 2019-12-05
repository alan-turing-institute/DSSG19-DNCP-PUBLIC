/*
 Creates complaints table.
*/

drop table if exists semantic.complaints;
create table semantic.complaints as
select
	id_llamado,
	fecha_resolucion_proceso,
	_tipo_resultado
from
	raw.proceso_juridico;
