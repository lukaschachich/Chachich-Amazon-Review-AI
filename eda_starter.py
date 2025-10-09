import pandas as pd # To manipulate data
import seaborn as sns # To visualize data
import matplotlib.pyplot as plt # To manipulate graphics of data

df = pd.read_csv("fake-reviews.csv") # load the dataset

# Look at what the data looks like
# print(df.head()) # print the first 5 rows of the dataframe
# print(df.info()) # print the summary of the dataframe, including the data types and non-null counts
# print(df.isnull().sum()) # check for missing values
# print(df.describe()) # print the statistical summary of the dataframe (only for numerical columns)


# Creating new features/columns
df["char_length"] = df["text_"].apply(len) # create a new column with the character length of the review text
# .apply is used to apply the function 'len' to each element in the column

df["word_count"] = df["text_"].str.split().apply(len) # create a new column with the word count of the review text
# .str is used to access string methods for the whole column
# .split() splits the string into a list of words

print(df.groupby('label').size()) # print the count of each label (0 or 1)
print(df.groupby('category').size()) # print the count of each category

print(df.describe()) # print the statistical summary again to see the new columns
print(df)



# Visualizations

# plt.figure()
# sns.boxplot(x="char_length", y="label", data=df) # create a boxplot to visualize the distribution of character length by label
# plt.title("Boxplot of Character Length by Label")
# plt.xlabel("Character Length")
# plt.ylabel("Label")
# plt.show(block = False)

# plt.figure()
# sns.boxplot(x="word_count", y="label", data=df) # create a boxplot to visualize the distribution of word count by label
# plt.xlabel("Word Count") 
# plt.title("Boxplot of Word Count by Label")
# plt.ylabel("Label")
# plt.show(block = False)

plt.figure()
sns.histplot(data=df, x="char_length", hue="label", bins=500, kde=True, element="step") # create a histogram to visualize the distribution of character length by label
plt.xlim(0, 400) # zoom in on the x-axis to see the distribution better
plt.title("Character Length Distribution by Label")
plt.show(block = False)

# Notes:
# When looking at histograms we can see that CG reviews tend to be around 75-100 characters long.

plt.figure()
sns.histplot(data=df, x="word_count", hue="label", bins=50, kde=True, element="step") # create a histogram to visualize the distribution of word count by label
plt.xlim(0, 100) # zoom in on the x-axis to see the distribution better
plt.title("Word Count Distribution by Label")
plt.show(block = False)

# Notes:
# When looking at histograms we can see that CG reviews tend to be around 15-20 words long.

# plt.figure()
# sns.(x="category", data =df) # create a barplot to visualize the count of reviews by category
# plt.title("Count of Reviews by Category")
# plt.xlabel("Category")
# plt.ylabel("Count")
# plt.show()

plt.figure()
sns.histplot(data=df, x="rating", hue="label", bins=5, kde=False, element="step") # create a histogram to visualize the distribution of ratings by label
plt.title("Rating Distribution by Label")
plt.show()

# Notes: