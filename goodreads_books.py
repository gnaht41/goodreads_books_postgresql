from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace, to_date, monotonically_increasing_id, explode, split, date_format
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, regexp_replace, to_date, monotonically_increasing_id,
    explode, split,  when, trim, collect_list, coalesce, lit, count
)

# Khởi tạo phiên Spark với MongoDB và PostgreSQL
spark = SparkSession.builder \
    .appName("Goodreads Spark with MongoDB and PostgreSQL") \
    .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1,org.postgresql:postgresql:42.7.4") \
    .config("spark.mongodb.read.connection.uri", "mongodb://localhost:27017/goodreads_db.books") \
    .getOrCreate()

# Thiết lập chính sách phân tích cú pháp thời gian thành LEGACY để xử lý ngày tháng cũ
spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")

# Cài đặt mức độ log: Đây là phương thức để thiết lập mức độ log của Spark. Khi bạn đặt mức log, Spark sẽ chỉ hiển thị các thông tin có mức độ quan trọng ngang hoặc cao hơn mức log đã đặt.
spark.sparkContext.setLogLevel("INFO")


# Đọc dữ liệu từ MongoDB
df = spark.read \
    .format("mongo") \
    .option("uri", "mongodb://localhost:27017/goodreads_db.books") \
    .load()


# Xử lý cột publish_date
df = df.withColumn("cleaned_date", regexp_replace(
    col("Date"), "First published ", ""))

# Chuyển đổi cột cleaned_date với các định dạng khác nhau và xử lý giá trị null
date_formats = [
    "MMMM d, yyyy",
    "yyyy",
    "MMMM yyyy"
]

# Khởi tạo publish_date với giá trị null
df = df.withColumn("publish_date", lit(None))

# Lặp qua các định dạng ngày
for date_format_str in date_formats:
    df = df.withColumn("publish_date",
                       when(col("publish_date").isNull(), to_date(
                           col("cleaned_date"), date_format_str))
                       .otherwise(col("publish_date")))

# Xử lý giá trị null và chuyển đổi cột publish_date
df = df.withColumn("publish_date",
                   coalesce(col("publish_date"), lit("1900-01-01")))

# Chuyển đổi về kiểu dữ liệu date
df = df.withColumn("publish_date", to_date(col("publish_date"), "yyyy-MM-dd"))


# Xử lý dữ liệu có dạng "8,932,568" - Loại bỏ dấu phẩy
df = df.withColumn("Number of Ratings", regexp_replace(col("Number of Ratings"), ",", "")) \
       .withColumn("Reviews", regexp_replace(col("Reviews"), ",", "")) \
       .withColumn("Score", regexp_replace(col("Score"), ",", ""))

# Chuyển đổi kiểu dữ liệu sau khi loại bỏ dấu phẩy
df = df.withColumn("Pages", col("Pages").cast("int")) \
       .withColumn("Rating", col("Rating").cast("float")) \
       .withColumn("Number of Ratings", col("Number of Ratings").cast("int")) \
       .withColumn("Reviews", col("Reviews").cast("int")) \
       .withColumn("Score", col("Score").cast("int"))


# Bước 4: Xử lý trường hợp Description không hợp lệ
df = df.withColumn("Description",
                   F.when(col("Description").isNull() | (F.trim(col("Description"))
                          == ""), "No description available")  # Thay thế NaN và chuỗi rỗng
                   # Thay thế chỉ số
                    .when(col("Description").rlike("^[0-9]+$"), "No description available")
                   # Thay thế chuỗi chỉ có khoảng trắng
                    .when(col("Description").rlike("^[\\s]+$"), "No description available")
                   # Thay thế chuỗi không hợp lệ
                    .when(col("Description").rlike("^[0-9]+[a-zA-Z]+|[a-zA-Z]+[0-9]+$"), "No description available")
                    .otherwise(col("Description")))

# Bước 6: Xóa dấu nháy kép và ký tự không hợp lệ trong Description
df = df.withColumn("Description", regexp_replace(
    col("Description"), '"', ''))  # Xóa dấu nháy kép
df = df.withColumn("Description", regexp_replace(
    col("Description"), '[^a-zA-Z0-9\\s]', ''))  # Xóa ký tự không hợp lệ

# Bước 7: Xóa dữ liệu trùng lặp
df = df.dropDuplicates()

# Bước 8: Xóa cột _id nếu có
df = df.drop("_id")

# Hiển thị DataFrame sau khi xử lý
df.show(truncate=False)

# Kiểm tra và điền giá trị mặc định cho các giá trị null
df = df.na.fill({
    "Rank": 0,
    "Title": "No title",
    "Author": "no author",
    "Rating": 0.0,
    "Number of Ratings": 0,
    "Description": "No description available",
    "Reviews": 0,
    "Pages": 0,
    "Cover Type": "No cover type",
    "Score": 0.0,
    "Genres": "No genres"
})

# # Thay thế giá trị null trong cột Date bằng ngày hiện tại
df = df.withColumn("Date",
                   F.when(F.col("Date").isNull(), F.current_date()).otherwise(F.col("Date")))

# # Định dạng clean_date thành dạng "Published July 15, 2022" nếu null
df = df.withColumn("cleaned_date",
                   F.when(F.col("cleaned_date").isNull(),
                          F.concat(F.lit("Published "),
                                   F.date_format(F.current_date(), "MMMM dd, yyyy")))
                   .otherwise(F.col("cleaned_date")))

# Kiểm tra lại số lượng giá trị null sau khi điền giá trị
null_counts_after = df.select(
    [count(when(col(c).isNull(), c)).alias(c) for c in df.columns])
null_counts_after.show(truncate=False)
