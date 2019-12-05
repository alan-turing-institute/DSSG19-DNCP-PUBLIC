# Data cleaning and features log

## Cleaned schema
### Getting and cleaning initial values

DNCP has indicated that there are certain variables that are available right at the start when the tender is received from the public agencies when they submit their tenders to DNCP.

We filtered the list obtain all the variables available from each of the tables.
  Criteria:
    1. Available at the start: Yes
    2. Can change over the process: No
Some of the data we received can change over time and are replaced over time. We do not want these variables to train our models as they represent time leakage. We can only predict over variables that tenders have at the beginning of the procurement process.

Variables and initial values for each tables are:

| Process | Documents | Contract | Complaint | Requested Item | Bidder | Catalogue | Tender verification history |  |  |  |  |  |  |  |  |  |  |  |  |
|---------------------------------|---------------------|---------------------------|---------------------|--------------------|----------------|----------------|-----------------------------|---|---|---|---|---|---|---|---|---|---|---|---|
| id_llamado | id_llamado | id_llamado | id_llamado | id_llamado | ruc_completo | nivel_1 | id |  |  |  |  |  |  |  |  |  |  |  |  |
| planificacion_slug | convocatoria_slug | planificacion_slug | id | convocatoria_slug | razon_social | descripcion_n1 | id_llamado |  |  |  |  |  |  |  |  |  |  |  |  |
| convocatoria_slug | id | convocatoria_slug | _tipo | _moneda | nro_licitacion | nivel_2 | final_state |  |  |  |  |  |  |  |  |  |  |  |  |
| nro_licitacion | nombre_archivo | adjudicacion_slug | tipo | moneda | llamado_id | descripcion_n2 | audit_user |  |  |  |  |  |  |  |  |  |  |  |  |
| convocante_slug | fecha_archivo | nro_licitacion | _motivo | id_item_solicitado | id | nivel_3 | audit_date |  |  |  |  |  |  |  |  |  |  |  |  |
| convocante_codigo | tipo_documento | nombre_licitacion | motivo |  |  | descripcion_n3 | reception_date |  |  |  |  |  |  |  |  |  |  |  |  |
| convocante | modulo | convocante_slug | denunciante |  |  | nivel_4 | time_taken |  |  |  |  |  |  |  |  |  |  |  |  |
| unidad_contratacion | tipo_documento_id | convocante_codigo | origen |  |  | descripcion_n4 |  |  |  |  |  |  |  |  |  |  |  |  |  |
| tipo_entidad | url | convocante | convocatoria_slug |  |  | nivel_5 |  |  |  |  |  |  |  |  |  |  |  |  |  |
| categoria_id | id_pbc_carta_adenda | unidad_contratacion | id_proceso_juridico |  |  | descripcion_n5 |  |  |  |  |  |  |  |  |  |  |  |  |  |
| categoria_codigo |  | tipo_entidad |  |  |  | id |  |  |  |  |  |  |  |  |  |  |  |  |  |
| categoria |  | categoria_id |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| _objeto_licitacion |  | categoria_codigo |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| fecha_publicacion_planificacion |  | categoria |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| _moneda |  | tipo_procedimiento_id |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| moneda |  | tipo_procedimiento_codigo |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| planificacion_url |  | tipo_procedimiento |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  | _sistema_adjudicacion |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  | sistema_adjudicacion |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  | proveedor |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  | ruc |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  | _moneda |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  | moneda |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  | id_contrato |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

### Proceso table
There are a few issues to highlight when producing the cleaned proceso table:
1. *tipo_procedimiento* and *tipo_procedimiento_codigo* do not map exactly. *tipo_procedimiento_codigo* has 15 unique categories whereas *tipo_procedimiento* has 34 categories.
  **Assumption:** We are only using 6 unique categories based on *tipo_procedimiento_codigo*:
  - Contratacion Directas (CD)
  - Concursos de Ofertas (CO)
  - Licitacion Publica Nacional (LPN)
  - Contratacion por Excepcion (CE)
  - Locacion de Inmeubles (LC)
  - Licitacion Publica Internacional (LPI)

2. **Assumption:** Monetary value is currently labeled as a possible variable that could change over the course of the process. However, after speaking with DNCP we know that this is an important variable to be considered at the start. So we are including the monetary value of tenders at publication.

### Tenders_record
1. Only extracted the reception date of the tender document. This is the date when the tender is first received. There are other variables such as the id of reviewer but these information should not have been know at the start even though it's been marked by DNCP as known beforehand.

### Oferentes
1. Use lower cased *razon_social* and rename it as *bidder_name*
2. Use lower cased *nombre_fantasia* and rename it as *bidder_commercial_name*
3. Create a *bidder_unique_identifier* by concatenating *ruc_completo* and lower cased *razon_social*

### Items
1. Clean the NULL values in the *presentacion* column by replacing them by \_NO_APLICA.

## Semantic Schema

### Tenders table
- Tenders table takes the variables from the cleaned proceso table ('proceso_initial_cleaned').In additional, it takes the reception date from the tenders_record table from cleaned schema.

- Out of the 125k tender records, around 800 are unable to match *reception_date* from the tender_records table. **Assumption:** We remove those tender records that do not have a *reception_date*.

### Features table
There are `agencies`, `agency_type`, `product_type`, `service_type`, `procurement_type` tables in the semantic schema. Each of these table relates to the 5 classifications we can give to a new tender. For example, for each new tender, we will know which agency is doing the procurement, whether the agency is a municipality etc. Each of these classifications will have historical data associated with them. We have the following list of features created for each of these classification tables:

| Variable type 	| Feature name                                  	| Description                                                                 	|
|---------------	|-----------------------------------------------	|-----------------------------------------------------------------------------	|
| Boolean       	| municipality_bool                             	| Boolean whether it is a municipality or not.                                	|
| Integer       	| n_tenders                                     	| Number of tenders received up to some date.                                 	|
| Float         	| avg_log_planned_real_value                    	|                                                                             	|
| Float         	| median_log_planned_real_value                 	|                                                                             	|
| Float         	| skew_log_planned_real_value                   	|                                                                             	|
| Float         	| kurt_log_planned_real_value                   	|                                                                             	|
| Float         	| quarter_percentil_log_planned_real_value      	|                                                                             	|
| Float         	| seventy_five_percentil_log_planned_real_value 	|                                                                             	|
| Float         	| two_percentil_log_planned_real_value          	|                                                                             	|
| Float         	| ninety_eight_log_planned_real_value           	|                                                                             	|
| Float         	| avg_log_real_value                            	| Actual values of tenders                                                    	|
| Float         	| median_log_real_value                         	| Actual values of tenders                                                    	|
| Float         	| skew_log_real_value                           	| Actual values of tenders                                                    	|
| Float         	| kurt_log_real_value                           	| Actual values of tenders                                                    	|
| Float         	| 25%_log_real_value                            	| Actual values of tenders                                                    	|
| Float         	| 75%_log_real_value                            	| Actual values of tenders                                                    	|
| Float         	| 2%_log_real_value                             	| Actual values of tenders                                                    	|
| Float         	| 98%_log_real_value                            	| Actual values of tenders                                                    	|
| Boolean       	| top_5_complained                              	| Is it an  agency that it in the top 5 agencies that had complained?         	|
| Boolean       	| top_10_complained                             	| Is it an  agency that it in the top 10 agencies that had complained?        	|
| Boolean       	| top_20_complained                             	| Is it an  agency that it in the top 20 agencies that had complained?        	|
| Integer       	| n_complaints                                  	| Number of complaints received (effective and uneffective).                  	|
| Integer       	| n_eff_complaints                              	| Number of effective complaints received.                                    	|
| Boolean       	| eff_complaints_bool                           	| Whether the agency ever received an effective complaint                     	|
| Integer       	| n_cancelled_complaints                        	| Number of cancelled complaints.                                             	|
| Integer       	| n_tenders_complaints                          	| Number of tenders with complaints.                                          	|
| Integer       	| n_tenders_eff_complaints                      	| Number of tenders with effective complaints.                                	|
| Integer       	| n_complaints_3m                               	| Number of complaints in the last 6 months.                                  	|
| Integer       	| n_eff_complaints_3m                           	| Number of effective complaints in the last 6 months.                        	|
| Integer       	| n_tenders_complaints_3m                       	| Number of complaints in the last 6 months.                                  	|
| Integer       	| n_tenders_eff_complaints_3m                   	| Number of effective complaints in the last 6 months.                        	|
| Float         	| avg_n_bidders                                 	| Average number of bidders                                                   	|
| Float         	| total_n_bidders                               	| Total number of bidders                                                     	|
| Float         	| median_n_bidders                              	| Median number of bidders                                                    	|
| Float         	| unique_number_products                        	| Number of unique products                                                   	|
| Float         	| mean_number_products                          	| Mean of distribution of number of products for the range of tenders         	|
| Float         	| median_number_products                        	| Median of distribution of number of products for the range of tenders       	|
| Float         	| p25_number_products                           	| Top 25% in the distribution of number of products for the range of tenders  	|
| Float         	| p75_number_products                           	| Top 75% in the distribution of number of products for the range of tenders  	|
