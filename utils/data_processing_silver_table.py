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


def process_silver_table(snapshot_date_str, bronze_lms_directory, silver_loan_daily_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_loan_daily_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_lms_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # augment data: add month on book
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

    # augment data: add days past due
    df = df.withColumn("installments_missed", F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
    df = df.withColumn("first_missed_date", F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
    df = df.withColumn("dpd", F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))

    # save silver table - IRL connect to database to write
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df

def process_silver_financials_table(snapshot_date_str, bronze_financials_directory, silver_financials_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # connect to bronze table
    partition_name = "bronze_financials_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_financials_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: some numeric columns have junk like a trailing "_", strip non-numeric chars then cast to float
    numeric_cols = ["Annual_Income", "Monthly_Inhand_Salary", "Num_of_Delayed_Payment", "Changed_Credit_Limit",
                    "Outstanding_Debt", "Credit_Utilization_Ratio", "Total_EMI_per_month",
                    "Amount_invested_monthly", "Monthly_Balance"]
    for column in numeric_cols:
        df = df.withColumn(column, F.regexp_replace(col(column).cast(StringType()), "[^0-9.-]", ""))
        df = df.withColumn(column, col(column).cast(FloatType()))

    # clean data: integer columns have some impossible outliers, cast to int and null out values outside sensible ranges
    df = df.withColumn("Num_Bank_Accounts", col("Num_Bank_Accounts").cast(IntegerType()))
    df = df.withColumn("Num_Bank_Accounts", F.when((col("Num_Bank_Accounts") < 0) | (col("Num_Bank_Accounts") > 20), None).otherwise(col("Num_Bank_Accounts")))

    df = df.withColumn("Num_Credit_Card", col("Num_Credit_Card").cast(IntegerType()))
    df = df.withColumn("Num_Credit_Card", F.when((col("Num_Credit_Card") < 0) | (col("Num_Credit_Card") > 15), None).otherwise(col("Num_Credit_Card")))

    df = df.withColumn("Interest_Rate", col("Interest_Rate").cast(IntegerType()))
    df = df.withColumn("Interest_Rate", F.when((col("Interest_Rate") < 0) | (col("Interest_Rate") > 50), None).otherwise(col("Interest_Rate")))

    df = df.withColumn("Num_of_Loan", col("Num_of_Loan").cast(IntegerType()))
    df = df.withColumn("Num_of_Loan", F.when((col("Num_of_Loan") < 0) | (col("Num_of_Loan") > 15), None).otherwise(col("Num_of_Loan")))

    df = df.withColumn("Num_Credit_Inquiries", col("Num_Credit_Inquiries").cast(IntegerType()))

    # clean data: a few more numeric columns have an injected garbage tail past the plausible range
    # (checked the percentiles - e.g. Total_EMI p95 is ~580 but p99 jumps to ~58000). null the impossible values.
    df = df.withColumn("Num_Credit_Inquiries", F.when((col("Num_Credit_Inquiries") < 0) | (col("Num_Credit_Inquiries") > 50), None).otherwise(col("Num_Credit_Inquiries")))
    df = df.withColumn("Num_of_Delayed_Payment", F.when((col("Num_of_Delayed_Payment") < 0) | (col("Num_of_Delayed_Payment") > 60), None).otherwise(col("Num_of_Delayed_Payment")))
    df = df.withColumn("Total_EMI_per_month", F.when((col("Total_EMI_per_month") < 0) | (col("Total_EMI_per_month") > 5000), None).otherwise(col("Total_EMI_per_month")))
    df = df.withColumn("Annual_Income", F.when((col("Annual_Income") < 0) | (col("Annual_Income") > 1000000), None).otherwise(col("Annual_Income")))
    df = df.withColumn("Monthly_Balance", F.when((col("Monthly_Balance") < -10000) | (col("Monthly_Balance") > 1000000), None).otherwise(col("Monthly_Balance")))

    # clean data: a negative delay just means paid early, so floor it at 0
    df = df.withColumn("Delay_from_due_date", col("Delay_from_due_date").cast(IntegerType()))
    df = df.withColumn("Delay_from_due_date", F.when(col("Delay_from_due_date") < 0, 0).otherwise(col("Delay_from_due_date")))

    # clean data: replace junk categorical values with "Unknown" (missingness can still be a signal)
    df = df.withColumn("Credit_Mix", F.when(col("Credit_Mix") == "_", "Unknown").otherwise(col("Credit_Mix")))
    valid_behaviour = ["High_spent_Small_value_payments", "High_spent_Medium_value_payments",
                       "High_spent_Large_value_payments", "Low_spent_Small_value_payments",
                       "Low_spent_Medium_value_payments", "Low_spent_Large_value_payments"]
    df = df.withColumn("Payment_Behaviour", F.when(col("Payment_Behaviour").isin(valid_behaviour), col("Payment_Behaviour")).otherwise("Unknown"))

    # augment data: Credit_History_Age is text like "10 Years and 9 Months", turn it into total months
    df = df.withColumn("credit_history_age_months",
                       (F.regexp_extract(col("Credit_History_Age"), "(\\d+) Years", 1).cast(IntegerType()) * 12
                        + F.regexp_extract(col("Credit_History_Age"), "(\\d+) Months", 1).cast(IntegerType())))

    # augment data: Type_of_Loan is a multi value string, count how many loan types are listed
    df = df.withColumn("num_loan_types",
                       F.when(col("Type_of_Loan").isNull(), 0)
                       .otherwise(F.size(F.split(F.regexp_replace(col("Type_of_Loan"), " and ", ", "), ", "))))

    # clean data: enforce schema / data type on remaining columns
    df = df.withColumn("Customer_ID", col("Customer_ID").cast(StringType()))
    df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))

    # save silver table - IRL connect to database to write
    partition_name = "silver_financials_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_financials_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)

    return df


def process_silver_attributes_table(snapshot_date_str, bronze_attributes_directory, silver_attributes_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # connect to bronze table
    partition_name = "bronze_attributes_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_attributes_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: Age has junk chars and impossible values, strip non-numeric then cast to int
    df = df.withColumn("Age", F.regexp_replace(col("Age").cast(StringType()), "[^0-9-]", ""))
    df = df.withColumn("Age", col("Age").cast(IntegerType()))
    df = df.withColumn("Age", F.when((col("Age") < 18) | (col("Age") > 100), None).otherwise(col("Age")))

    # clean data: Occupation has junk like "_______", replace all-underscore values with "Unknown"
    df = df.withColumn("Occupation", F.when(col("Occupation").rlike("^_+$"), "Unknown").otherwise(col("Occupation")))

    # clean data: drop Name and SSN - they are personal identifiers with no predictive value (and SSN is mostly junk anyway)
    df = df.drop("Name", "SSN")

    # clean data: enforce schema / data type
    df = df.withColumn("Customer_ID", col("Customer_ID").cast(StringType()))
    df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))

    # save silver table - IRL connect to database to write
    partition_name = "silver_attributes_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_attributes_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)

    return df


def process_silver_clickstream_table(snapshot_date_str, bronze_clickstream_directory, silver_clickstream_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # connect to bronze table
    partition_name = "bronze_clickstream_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_clickstream_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: the 20 clickstream features are already numeric, just enforce integer type
    for i in range(1, 21):
        column = "fe_" + str(i)
        df = df.withColumn(column, col(column).cast(IntegerType()))

    # clean data: enforce schema / data type
    df = df.withColumn("Customer_ID", col("Customer_ID").cast(StringType()))
    df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))

    # note: we keep clickstream at its natural customer-month grain here.
    # the as-of aggregation up to loan application date is done in the gold step.

    # save silver table - IRL connect to database to write
    partition_name = "silver_clickstream_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_clickstream_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)

    return df
