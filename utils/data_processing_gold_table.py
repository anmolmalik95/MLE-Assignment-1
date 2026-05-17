import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import argparse

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_labels_gold_table(snapshot_date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd, mob):
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to silver table
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    # get customer at mob
    df = df.filter(col("mob") == mob)

    # get label
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(str(dpd)+'dpd_'+str(mob)+'mob').cast(StringType()))

    # select columns to save
    df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")

    # save gold table - IRL connect to database to write
    partition_name = "gold_label_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_label_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df

def process_features_gold_table(snapshot_date_str, silver_loan_daily_directory, silver_financials_directory, silver_attributes_directory, silver_clickstream_directory, gold_feature_store_directory, spark, mob):

    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # connect to silver loan_daily for this snapshot - this gives us the loans we build features for
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    loans = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', loans.count())

    # get customer at mob (same population as the label store, so they line up 1 to 1)
    loans = loans.filter(col("mob") == mob)
    loans = loans.select("loan_id", "Customer_ID", "loan_start_date", "snapshot_date")

    # load all silver feature partitions (a customer's feature snapshot can be in any month)
    fin_files = [silver_financials_directory + os.path.basename(f) for f in glob.glob(os.path.join(silver_financials_directory, '*'))]
    att_files = [silver_attributes_directory + os.path.basename(f) for f in glob.glob(os.path.join(silver_attributes_directory, '*'))]
    clk_files = [silver_clickstream_directory + os.path.basename(f) for f in glob.glob(os.path.join(silver_clickstream_directory, '*'))]
    fin = spark.read.parquet(*fin_files)
    att = spark.read.parquet(*att_files)
    clk = spark.read.parquet(*clk_files)

    # ---- financials: join on customer, then null out anything not known at loan application date ----
    fin = fin.withColumnRenamed("snapshot_date", "fin_snapshot_date")
    df = loans.join(fin, on="Customer_ID", how="left")
    fin_feature_cols = ["Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts", "Num_Credit_Card",
                        "Interest_Rate", "Num_of_Loan", "Delay_from_due_date", "Num_of_Delayed_Payment",
                        "Changed_Credit_Limit", "Num_Credit_Inquiries", "Credit_Mix", "Outstanding_Debt",
                        "Credit_Utilization_Ratio", "Payment_of_Min_Amount", "Total_EMI_per_month",
                        "Amount_invested_monthly", "Payment_Behaviour", "Monthly_Balance",
                        "credit_history_age_months", "num_loan_types"]
    for column in fin_feature_cols:
        df = df.withColumn(column, F.when(col("fin_snapshot_date") <= col("loan_start_date"), col(column)).otherwise(None))
    df = df.drop("fin_snapshot_date")

    # ---- attributes: same as-of rule ----
    att = att.withColumnRenamed("snapshot_date", "att_snapshot_date")
    df = df.join(att, on="Customer_ID", how="left")
    att_feature_cols = ["Age", "Occupation"]
    for column in att_feature_cols:
        df = df.withColumn(column, F.when(col("att_snapshot_date") <= col("loan_start_date"), col(column)).otherwise(None))
    df = df.drop("att_snapshot_date")

    # ---- clickstream: aggregate only the months on or before the loan application date ----
    clk = clk.join(loans.select("loan_id", "Customer_ID", "loan_start_date"), on="Customer_ID", how="inner")
    clk = clk.filter(col("snapshot_date") <= col("loan_start_date"))
    agg_exprs = [F.avg("fe_" + str(i)).alias("fe_" + str(i) + "_avg") for i in range(1, 21)]
    agg_exprs = agg_exprs + [F.count(F.lit(1)).alias("clickstream_months")]
    clk_agg = clk.groupBy("loan_id").agg(*agg_exprs)
    df = df.join(clk_agg, on="loan_id", how="left")

    # keep only the curated feature columns (drop raw text cols that we already engineered into new ones)
    clickstream_cols = ["fe_" + str(i) + "_avg" for i in range(1, 21)] + ["clickstream_months"]
    keep_cols = ["loan_id", "Customer_ID", "loan_start_date", "snapshot_date"] + fin_feature_cols + att_feature_cols + clickstream_cols
    df = df.select(*keep_cols)

    # save gold feature store - IRL connect to database to write
    partition_name = "gold_feature_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_feature_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)

    return df
