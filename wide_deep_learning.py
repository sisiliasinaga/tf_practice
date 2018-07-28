import tensorflow as tf

#categorical base columns
gender = tf.contrib.layers.sparse_column_with_keys(column_name = "gender", keys = ["Female", "Male"])
race = tf.contrib.layers.sparse_column_with_keys(column_name = "race", keys = ["Amer-Indian-Eskimo", "Asian-Pac-Islander", "Black", "Other", "White"])
education = tf.contrib.layers.sparse_column_with_hash_bucket("educaation", hash_bucket_size = 1000)
relationship = tf.contrib.layers.sparse_column_with_hash_bucket("relationship", hash_bucket_size = 100)
workclass = tf.contrib.layers.sparse_column_with_hash_bucket("workclass", hash_bucket_size = 100)
occupation = tf.contrib.layers.sparse_column_with_hash_bucket("occupation", hash_bucket_size = 1000)
native_country = tf.contrib.layers.sparse_column_with_hash_bucket("native_country", hash_bucket_size = 1000)

#continuous base columns
age = tf.contrib.layers.real_valued_column("age")
age_buckets = tf.contrib.layers.bucketized_column(age, boundaries = [18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
education_num = tf.contrib.layers.real_valued_column("education_num")
capital_gain = tf.contrib.layers.real_valued_column("capital_gain")
capital_loss = tf.contrib.layers.real_valued_column("capital_loss")
hours_per_week = tf.contrib.layers.real_valued_column("hours_per_week")

wide_columns = [gender, native_country, education, occupation, workclass, relationship, age_buckets, tf.contrib.layers.crossed_column([education, occupation], hash_bucket_size = int(1e4)), tf.contrib.layers.crossed_column([native_country, occupation], hash_bucket_size = int(1e4)), tf.contrib.layers.crossed_column([age_buckets, education, occupation], hash_bucket_size = int(1e6))]
deep_columns = [tf.contrib.layers.embedding_column(workclass, dimension=8), tf.contrib.layers.embedding_column(education, dimensions = 8), tf.contrib.layers.embedding_column(gender, dimensions = 8), tf.contrib.layers.embedding_column(relationship, dimensions = 8), tf.contrib.layers.embedding_column(native_country, dimensions = 8), tf.contrib.layers.embedding_column(occupation, dimensions = 8), age, education_num, capital_gain, capital_loss, hours_per_week]