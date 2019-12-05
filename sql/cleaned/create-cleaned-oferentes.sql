-- Create oferentes tables in cleaned schema
/* Added an unique identifier of bidder
 * as some rucs are null and some bidders_name
 * are repeated but with different ruc number
 */

create cleaned.oferentes as (
  select
    ruc_completo as bidder_ruc
    , lower(razon_social) as bidder_name
    , lower(nombre_fantasia) as bidder_commercial_name
    , concat(ruc_completo,'_',lower(razon_social))as bidder_unique_identifier
    , nro_licitacion
    , id_llamado
)

alter table cleaned.oferentes primary key (id_llamado,bidder_unique_identifier)
