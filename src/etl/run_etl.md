# ETL Pipeline


The ETL process is built as follow

1. 01_Settingup_database contains instructions to copy original sql files sent by partner and use them to build database and create raw and semantic schemas
2. 02_create_tables_support_doc_schema creates schema and tables for currency convertion and inflation
3. 03_create_raw_label_schema creates schema with a table at tender level that contains an label that identifies if the tender received an effective complaint. This schema is used for data statistics.
4. 04_create_tables_semantic_schema creates tables in the semantic schemas

etl_run.sh executes the listed file in order.

Inputs needed for this process:
* Name of file that contain database dump file
* Name of file that contains tender_record table dump file
* Name to assign to the database

```python

```
